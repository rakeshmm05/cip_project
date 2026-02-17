import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("../dataset/balanced_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

WINDOW_SECONDS = 5

# ==========================================================
# 2. BUILD SINGLE GRAPH SNAPSHOT FUNCTION
# ==========================================================

def build_graph_snapshot(window_df):

    xapps = window_df["xapp_id"].unique()
    ues = window_df["ue_id"].unique()

    xapp_index = {x:i for i,x in enumerate(xapps)}
    ue_index = {u:i+len(xapps) for i,u in enumerate(ues)}

    num_nodes = len(xapps) + len(ues)

    node_features = torch.zeros((num_nodes, 6))

    # -------------------------------------------
    # XAPP NODE FEATURES
    # -------------------------------------------

    for xapp in xapps:
        subset = window_df[window_df["xapp_id"] == xapp]
        idx = xapp_index[xapp]

        node_features[idx,0] = subset["xapp_action_rate"].mean()
        node_features[idx,1] = subset["message_frequency"].mean()
        node_features[idx,2] = subset["rolling_degradation_count"].mean()
        node_features[idx,3] = subset["unique_target_ratio"].mean()
        node_features[idx,4] = subset["context_violation_score"].mean()
        node_features[idx,5] = subset["persistent_target_flag"].max()

    # -------------------------------------------
    # UE NODE FEATURES
    # -------------------------------------------

    for ue in ues:
        subset = window_df[window_df["ue_id"] == ue]
        idx = ue_index[ue]

        node_features[idx,0] = subset["rsrp"].mean()
        node_features[idx,1] = subset["sinr"].mean()
        node_features[idx,2] = subset["interference_level"].mean()
        node_features[idx,3] = subset["cell_load"].mean()
        node_features[idx,4] = subset["latency_ms"].mean()
        node_features[idx,5] = subset["throughput_mbps"].mean()

    # -------------------------------------------
    # EDGE INDEX + FEATURES
    # -------------------------------------------

    edge_index = []
    edge_attr = []

    for _, row in window_df.iterrows():

        src = xapp_index[row["xapp_id"]]
        dst = ue_index[row["ue_id"]]

        edge_index.append([src, dst])

        edge_attr.append([
            row["action_context_match"],
            row["performance_delta"],
            row["degradation_flag"],
            row["repeat_target_count"],
            row["context_violation_score"],
            row["persistent_target_flag"]
        ])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # -------------------------------------------
    # LABEL (xApp-level)
    # -------------------------------------------

    labels = torch.zeros(num_nodes)

    for xapp in xapps:
        subset = window_df[window_df["xapp_id"] == xapp]
        if subset["is_malicious"].max() == 1:
            labels[xapp_index[xapp]] = 1

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    )

# ==========================================================
# 3. BUILD ALL SNAPSHOTS
# ==========================================================

graphs = []

start_time = df["timestamp"].min()
end_time = df["timestamp"].max()

current_time = start_time

while current_time < end_time:

    window_df = df[
        (df["timestamp"] >= current_time) &
        (df["timestamp"] < current_time + pd.Timedelta(seconds=WINDOW_SECONDS))
    ]

    if len(window_df) > 0:
        graph = build_graph_snapshot(window_df)
        graphs.append(graph)

    current_time += pd.Timedelta(seconds=WINDOW_SECONDS)

print("Total graph snapshots:", len(graphs))
