import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time

# ==========================================================
# 1. GLOBAL CONFIGURATION
# ==========================================================

SEED = 42
np.random.seed(SEED)

NUM_XAPPS = 4
NUM_UES = 60
NUM_CELLS = 5
TOTAL_EVENTS = 60000
TIME_STEP_MS = 100

ROGUE_XAPPS = ["xapp_2", "xapp_3"]
TARGET_UES_PER_ROGUE = 8

PERSISTENCE_WINDOW = 15
PERSISTENCE_THRESHOLD = 4

TARGET_PER_CLASS = TOTAL_EVENTS // 2

# ==========================================================
# 2. ROGUE TARGET ASSIGNMENT
# ==========================================================

rogue_targets = {
    rx: np.random.choice(range(1, NUM_UES + 1),
                         TARGET_UES_PER_ROGUE,
                         replace=False)
    for rx in ROGUE_XAPPS
}

# ==========================================================
# 3. PHYSICS-CONSISTENT RADIO MODEL
# ==========================================================

def generate_radio_state(cell_load):
    rsrp = np.random.normal(-90, 6)
    interference = np.clip(np.random.normal(8 + 0.05 * cell_load, 2), 0, 30)
    sinr = np.clip(20 - interference + np.random.normal(0, 2), -5, 30)

    bandwidth = 20
    throughput = bandwidth * np.log2(1 + max(sinr, 0))

    latency = np.clip(40 - 0.8 * sinr + 0.4 * cell_load, 5, 200)
    packet_loss = np.clip(0.01 * interference + np.random.normal(0, 0.01), 0, 0.2)

    return rsrp, sinr, interference, throughput, latency, packet_loss

def expected_policy(rsrp, sinr, cell_load):
    if rsrp < -105 or sinr < 5:
        return "power_increase"
    elif cell_load > 85:
        return "handover"
    else:
        return "no_action"

# ==========================================================
# 4. DATA GENERATION
# ==========================================================

start_time = time.time()

records = []

xapp_last_time = {}
xapp_action_history = defaultdict(lambda: deque())
target_history = defaultdict(lambda: deque())
degradation_history = defaultdict(lambda: deque(maxlen=PERSISTENCE_WINDOW))

timestamps = pd.date_range(
    start="2026-01-01",
    periods=TOTAL_EVENTS,
    freq=f"{TIME_STEP_MS}ms"
)

malicious_count = 0
benign_count = 0

for idx, ts in enumerate(timestamps):

    force_malicious = malicious_count < TARGET_PER_CLASS

    if force_malicious:
        xapp_id = np.random.choice(ROGUE_XAPPS)
        ue_id = np.random.choice(rogue_targets[xapp_id])
    else:
        xapp_id = f"xapp_{np.random.randint(0, NUM_XAPPS)}"
        ue_id = np.random.randint(1, NUM_UES + 1)

    cell_id = np.random.randint(1, NUM_CELLS + 1)
    neighbor_cell_count = np.random.randint(2, 6)
    ue_mobility_speed = np.random.uniform(0, 80)
    ue_session_duration = np.random.uniform(0, 600)

    cell_load = np.clip(np.random.normal(65, 10), 10, 100)
    active_ue_count = int(cell_load * 2)
    prb_utilization = cell_load + np.random.normal(0, 3)

    rsrp, sinr, interference, throughput, latency, packet_loss = \
        generate_radio_state(cell_load)

    expected_action = expected_policy(rsrp, sinr, cell_load)

    # ======================================================
    # Action Decision
    # ======================================================

    if force_malicious:
        action_type = "handover"
    else:
        action_type = expected_action

    target_cell_id = np.random.randint(1, NUM_CELLS + 1) \
        if action_type == "handover" else 0

    power_delta = 2 if action_type == "power_increase" else 0
    handover_flag = int(action_type == "handover")
    resource_block_delta = np.random.randint(-5, 6)

    # ======================================================
    # Temporal Features (Efficient Sliding Window)
    # ======================================================

    time_delta_prev = (
        (ts - xapp_last_time[xapp_id]).total_seconds()
        if xapp_id in xapp_last_time else 0
    )

    xapp_last_time[xapp_id] = ts

    action_history = xapp_action_history[xapp_id]
    action_history.append(ts)

    while action_history and \
          (ts - action_history[0]).total_seconds() > 5:
        action_history.popleft()

    xapp_action_rate = len(action_history) / 5.0
    message_frequency = len(action_history)

    # ======================================================
    # Context & Performance
    # ======================================================

    action_context_match = int(action_type == expected_action)

    if force_malicious:
        performance_delta = -3
        degradation_flag = 1
    else:
        performance_delta = 2
        degradation_flag = 0

    # ======================================================
    # Relational Persistence (Efficient)
    # ======================================================

    key = (xapp_id, ue_id)

    target_hist = target_history[key]
    target_hist.append(ts)

    while target_hist and \
          (ts - target_hist[0]).total_seconds() > 5:
        target_hist.popleft()

    repeat_target_count = len(target_hist)

    degradation_history[key].append(degradation_flag)
    rolling_degradation_count = sum(degradation_history[key])

    persistent_target_flag = int(
        rolling_degradation_count >= PERSISTENCE_THRESHOLD
    )

    context_violation_score = (
        (1 - action_context_match)
        * (rolling_degradation_count / PERSISTENCE_WINDOW)
    )

    is_malicious = int(force_malicious)

    if is_malicious:
        malicious_count += 1
    else:
        benign_count += 1

    unique_target_ratio = 0.5
    flow_id = f"{xapp_id}_{ue_id}_{idx}"

    records.append([
        ts, time_delta_prev, xapp_id, xapp_action_rate,
        ue_id, ue_mobility_speed, ue_session_duration,
        cell_id, neighbor_cell_count,
        action_type, target_cell_id,
        power_delta, handover_flag, resource_block_delta,
        rsrp, sinr, interference,
        cell_load, active_ue_count, prb_utilization,
        latency, packet_loss, throughput,
        expected_action, action_context_match,
        performance_delta, degradation_flag,
        message_frequency, repeat_target_count,
        unique_target_ratio, flow_id,
        rolling_degradation_count,
        persistent_target_flag,
        context_violation_score,
        is_malicious
    ])

# ==========================================================
# 5. EXPORT
# ==========================================================

columns = [
    "timestamp","time_delta_prev","xapp_id","xapp_action_rate",
    "ue_id","ue_mobility_speed","ue_session_duration",
    "cell_id","neighbor_cell_count","action_type","target_cell_id",
    "power_delta","handover_flag","resource_block_delta",
    "rsrp","sinr","interference_level",
    "cell_load","active_ue_count","prb_utilization",
    "latency_ms","packet_loss_rate","throughput_mbps",
    "expected_action","action_context_match",
    "performance_delta","degradation_flag",
    "message_frequency","repeat_target_count",
    "unique_target_ratio","flow_id",
    "rolling_degradation_count",
    "persistent_target_flag",
    "context_violation_score",
    "is_malicious"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("balanced_dataset.csv", index=False)

end_time = time.time()

print("Dataset generated successfully.")
print("Total samples:", len(df))
print("Malicious:", malicious_count)
print("Benign:", benign_count)
print("Malicious ratio:", df["is_malicious"].mean())
print("Execution time (seconds):", round(end_time - start_time, 3))
