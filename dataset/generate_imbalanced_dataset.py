import numpy as np
import pandas as pd
from collections import defaultdict, deque

# ==========================================================
# 1. GLOBAL CONFIGURATION
# ==========================================================

SEED = 42
np.random.seed(SEED)

NUM_XAPPS = 4
NUM_UES = 60
NUM_CELLS = 5
TOTAL_EVENTS = 20000
TIME_STEP_MS = 100

ROGUE_XAPPS = ["xapp_2", "xapp_3"]
TARGET_UES_PER_ROGUE = 6

PERSISTENCE_WINDOW = 15
PERSISTENCE_THRESHOLD = 4

# ==========================================================
# 2. ROGUE TARGET ASSIGNMENT
# ==========================================================

rogue_targets = {
    rx: set(np.random.choice(range(1, NUM_UES + 1),
                             TARGET_UES_PER_ROGUE,
                             replace=False))
    for rx in ROGUE_XAPPS
}

# ==========================================================
# 3. RADIO MODEL
# ==========================================================

def generate_radio_state(cell_load):
    rsrp = np.random.normal(-90, 6)
    interference = np.clip(np.random.normal(8 + 0.05 * cell_load, 2), 0, 30)
    sinr = np.clip(20 - interference + np.random.normal(0, 2), -5, 30)

    throughput = 20 * np.log2(1 + max(sinr, 0))
    latency = np.clip(40 - 0.8 * sinr + 0.4 * cell_load, 5, 200)
    packet_loss = np.clip(0.01 * interference + np.random.normal(0, 0.01), 0, 0.2)

    return rsrp, sinr, interference, throughput, latency, packet_loss


def expected_policy(rsrp, sinr, cell_load):
    if rsrp < -105 or sinr < 5:
        return "power_increase"
    elif cell_load > 85:
        return "handover"
    return "no_action"

# ==========================================================
# 4. DATA GENERATION (OPTIMIZED)
# ==========================================================

records = []

# Faster structures
xapp_last_time = {}
xapp_action_history = defaultdict(lambda: deque())
target_history = defaultdict(lambda: deque())
degradation_history = defaultdict(lambda: deque(maxlen=PERSISTENCE_WINDOW))

# Track unique targets per xapp efficiently
xapp_unique_targets = defaultdict(set)

current_time = 0  # milliseconds

for idx in range(TOTAL_EVENTS):

    current_time += TIME_STEP_MS

    xapp_id = f"xapp_{np.random.randint(NUM_XAPPS)}"
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

    # Rogue behavior
    if xapp_id in ROGUE_XAPPS and ue_id in rogue_targets[xapp_id]:
        action_type = "handover"
    else:
        action_type = expected_action

    target_cell_id = np.random.randint(1, NUM_CELLS + 1) \
        if action_type == "handover" else 0

    power_delta = 2 if action_type == "power_increase" else 0
    handover_flag = int(action_type == "handover")
    resource_block_delta = np.random.randint(-5, 6)

    # ------------------------------------------------------
    # Temporal features (FAST sliding window)
    # ------------------------------------------------------

    time_delta_prev = current_time - xapp_last_time.get(xapp_id, current_time)
    xapp_last_time[xapp_id] = current_time

    history = xapp_action_history[xapp_id]
    history.append(current_time)

    while history and current_time - history[0] > 5000:
        history.popleft()

    message_frequency = len(history)
    xapp_action_rate = message_frequency / 5.0

    # ------------------------------------------------------
    # Context consistency
    # ------------------------------------------------------

    action_context_match = int(action_type == expected_action)

    performance_delta = 0
    degradation_flag = 0

    if action_type == "handover" and expected_action != "handover":
        performance_delta = -3
        degradation_flag = 1
    elif action_type == "power_increase":
        performance_delta = 2

    # ------------------------------------------------------
    # Relational features (FAST)
    # ------------------------------------------------------

    key = (xapp_id, ue_id)

    th = target_history[key]
    th.append(current_time)

    while th and current_time - th[0] > 5000:
        th.popleft()

    repeat_target_count = len(th)

    xapp_unique_targets[xapp_id].add(ue_id)
    unique_target_ratio = len(xapp_unique_targets[xapp_id]) / max(1, message_frequency)

    dh = degradation_history[key]
    dh.append(degradation_flag)
    rolling_degradation_count = sum(dh)

    persistent_target_flag = int(
        rolling_degradation_count >= PERSISTENCE_THRESHOLD
        and xapp_id in ROGUE_XAPPS
        and ue_id in rogue_targets[xapp_id]
    )

    context_violation_score = (
        (1 - action_context_match) * (rolling_degradation_count / PERSISTENCE_WINDOW)
    )

    is_malicious = persistent_target_flag

    records.append([
        current_time,
        time_delta_prev,
        xapp_id,
        xapp_action_rate,
        ue_id,
        ue_mobility_speed,
        ue_session_duration,
        cell_id,
        neighbor_cell_count,
        action_type,
        target_cell_id,
        power_delta,
        handover_flag,
        resource_block_delta,
        rsrp,
        sinr,
        interference,
        cell_load,
        active_ue_count,
        prb_utilization,
        latency,
        packet_loss,
        throughput,
        expected_action,
        action_context_match,
        performance_delta,
        degradation_flag,
        message_frequency,
        repeat_target_count,
        unique_target_ratio,
        f"{xapp_id}_{ue_id}_{idx}",
        rolling_degradation_count,
        persistent_target_flag,
        context_violation_score,
        is_malicious
    ])

# ==========================================================
# 5. EXPORT
# ==========================================================

columns = [
    "timestamp",
    "time_delta_prev",
    "xapp_id",
    "xapp_action_rate",
    "ue_id",
    "ue_mobility_speed",
    "ue_session_duration",
    "cell_id",
    "neighbor_cell_count",
    "action_type",
    "target_cell_id",
    "power_delta",
    "handover_flag",
    "resource_block_delta",
    "rsrp",
    "sinr",
    "interference_level",
    "cell_load",
    "active_ue_count",
    "prb_utilization",
    "latency_ms",
    "packet_loss_rate",
    "throughput_mbps",
    "expected_action",
    "action_context_match",
    "performance_delta",
    "degradation_flag",
    "message_frequency",
    "repeat_target_count",
    "unique_target_ratio",
    "flow_id",
    "rolling_degradation_count",
    "persistent_target_flag",
    "context_violation_score",
    "is_malicious"
]

df = pd.DataFrame(records, columns=columns)
df.to_csv("imbalanced_dataset.csv", index=False)

print("Dataset generated successfully.")
print("Total samples:", len(df))
print("Malicious samples:", df["is_malicious"].sum())
print("Malicious ratio:", df["is_malicious"].mean())
