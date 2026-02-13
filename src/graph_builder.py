import pandas as pd
import torch
from torch_geometric.data import Data

from src.temporal import temporal_encoding

WINDOW = 20


def build_graph_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Encode IDs into a shared node index space: xApps first, then UEs.
    xapp_ids = {id_: i for i, id_ in enumerate(df["xapp_id"].unique())}
    ue_ids = {id_: i + len(xapp_ids) for i, id_ in enumerate(df["ue_id"].unique())}
    num_nodes = len(xapp_ids) + len(ue_ids)

    # Rolling statistics containers
    xapp_stats = {i: [] for i in xapp_ids.values()}
    ue_stats = {i: [] for i in ue_ids.values()}

    edge_index = []
    edge_attr = []
    labels = []
    time_values = []

    for idx, row in df.iterrows():
        src = xapp_ids[row["xapp_id"]]
        dst = ue_ids[row["ue_id"]]

        throughput = float(row["throughput_mbps"])
        xapp_stats[src].append(throughput)
        ue_stats[dst].append(throughput)

        if len(xapp_stats[src]) > WINDOW:
            xapp_stats[src].pop(0)
        if len(ue_stats[dst]) > WINDOW:
            ue_stats[dst].pop(0)

        xapp_mean = sum(xapp_stats[src]) / len(xapp_stats[src])
        ue_mean = sum(ue_stats[dst]) / len(ue_stats[dst])

        edge_index.append([src, dst])
        edge_attr.append(
            [
                float(row["rsrp"]),
                float(row["sinr"]),
                float(row["cell_load"]),
                throughput,
                float(row["latency_ms"]),
                xapp_mean,
                ue_mean,
            ]
        )
        labels.append(int(row["is_malicious"]))
        time_values.append(idx)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    time_tensor = torch.tensor(time_values, dtype=torch.float)
    denom = (time_tensor.max() - time_tensor.min()).clamp(min=1.0)
    norm_time = (time_tensor - time_tensor.min()) / denom
    temporal_features = temporal_encoding(norm_time)

    # Placeholder deterministic node features; model relies on edges.
    x = torch.zeros((num_nodes, 16), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    data.temporal = temporal_features
    return data


def build_window_graph_from_csv(
    csv_path,
    window_seconds=2,
    malicious_min_degradations=3,
):
    """
    Build relation-level graph samples at (window, xApp, UE) granularity.
    Each edge is one aggregated relation instance in a time window.
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    xapp_ids = {id_: i for i, id_ in enumerate(df["xapp_id"].unique())}
    ue_ids = {id_: i + len(xapp_ids) for i, id_ in enumerate(df["ue_id"].unique())}
    num_nodes = len(xapp_ids) + len(ue_ids)

    # 2-second (configurable) window buckets.
    base = df["timestamp"].min()
    win_delta = pd.to_timedelta(window_seconds, unit="s")
    df["window_idx"] = ((df["timestamp"] - base) // win_delta).astype(int)

    group_cols = ["window_idx", "xapp_id", "ue_id"]
    grouped = df.groupby(group_cols, sort=True)

    edge_index = []
    edge_attr = []
    labels = []
    time_values = []

    for (window_idx, xapp_id, ue_id), g in grouped:
        src = xapp_ids[xapp_id]
        dst = ue_ids[ue_id]

        deg_count = int(g["degradation_flag"].sum())
        context_violations = int(g["context_violation_score"].sum())
        malicious_label = int(deg_count >= malicious_min_degradations)

        edge_index.append([src, dst])
        edge_attr.append(
            [
                float(g["rsrp"].mean()),
                float(g["sinr"].mean()),
                float(g["cell_load"].mean()),
                float(g["throughput_mbps"].mean()),
                float(g["latency_ms"].mean()),
                float(g["throughput_mbps"].std(ddof=0)),
                float(g["latency_ms"].std(ddof=0)),
                float(deg_count),
                float(context_violations),
                float(len(g)),
            ]
        )
        labels.append(malicious_label)
        time_values.append(int(window_idx))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    time_tensor = torch.tensor(time_values, dtype=torch.float)
    denom = (time_tensor.max() - time_tensor.min()).clamp(min=1.0)
    norm_time = (time_tensor - time_tensor.min()) / denom
    temporal_features = temporal_encoding(norm_time)

    x = torch.zeros((num_nodes, 16), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    data.temporal = temporal_features
    return data
