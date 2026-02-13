import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================================
# 1. GLOBAL CONFIGURATION 
# ==========================================================

SEED = 42
np.random.seed(SEED)

NUM_XAPPS = 3
NUM_UES = 100
NUM_CELLS = 5
TOTAL_EVENTS = 20000
TIME_STEP_MS = 100

ROGUE_XAPP = "xapp_2"
TARGET_UES = [12, 45, 78]

PERSISTENCE_WINDOW = 20      # number of recent events to evaluate
PERSISTENCE_THRESHOLD = 3    # degradations needed to label malicious

# ==========================================================
# 2. PHYSICS-CONSISTENT RADIO MODEL
# ==========================================================

def generate_radio_state():
    """
    Generates telecom-realistic radio conditions.
    """
    rsrp = np.random.normal(-90, 8)             # typical LTE/5G range
    sinr = np.clip(np.random.normal(15, 5), -5, 30)
    cell_load = np.clip(np.random.normal(60, 15), 10, 100)

    # Throughput derived from Shannon-like behavior (simplified)
    bandwidth = 20  # MHz
    throughput = bandwidth * np.log2(1 + max(sinr, 0))  # Mbps approx

    # Latency inversely related to SINR & directly to load
    latency = np.clip(50 - sinr + 0.5 * cell_load, 5, 200)

    return rsrp, sinr, cell_load, throughput, latency


# ==========================================================
# 3. CONTEXT-DERIVED EXPECTED POLICY (GROUND TRUTH LOGIC)
# ==========================================================

def expected_action_policy(rsrp, sinr, cell_load):
    """
    Deterministic context-based decision logic.
    """
    if rsrp < -105 or sinr < 3:
        return "power_increase"
    elif cell_load > 85:
        return "handover"
    else:
        return "no_action"


# ==========================================================
# 4. XAPP BEHAVIOR MODELS
# ==========================================================

def benign_xapp(rsrp, sinr, cell_load):
    return expected_action_policy(rsrp, sinr, cell_load)

def rogue_xapp(ue_id, rsrp, sinr, cell_load):
    """
    Structured targeted degradation:
    Intermittently forces handover for selected UEs under good context.
    """
    if ue_id in TARGET_UES and rsrp > -95 and sinr > 10:
        return "handover"
    return expected_action_policy(rsrp, sinr, cell_load)


# ==========================================================
# 5. DATASET GENERATION
# ==========================================================

records = []
degradation_history = defaultdict(list)

timestamps = pd.date_range(
    start="2026-01-01",
    periods=TOTAL_EVENTS,
    freq=f"{TIME_STEP_MS}ms"
)

for i, ts in enumerate(timestamps):

    xapp_id = f"xapp_{np.random.randint(0, NUM_XAPPS)}"
    ue_id = np.random.randint(1, NUM_UES + 1)
    cell_id = np.random.randint(1, NUM_CELLS + 1)

    rsrp, sinr, cell_load, throughput, latency = generate_radio_state()

    expected_action = expected_action_policy(rsrp, sinr, cell_load)

    if xapp_id == ROGUE_XAPP:
        action = rogue_xapp(ue_id, rsrp, sinr, cell_load)
    else:
        action = benign_xapp(rsrp, sinr, cell_load)

    action_context_match = int(action == expected_action)

    # ======================================================
    # Performance impact modeling (physics-consistent)
    # ======================================================

    performance_delta = 0
    degradation_flag = 0

    if action == "power_increase":
        performance_delta = 2
    elif action == "handover":
        # If handover not required → slight degradation
        if expected_action != "handover":
            performance_delta = -3
        else:
            performance_delta = 1

    if performance_delta < 0:
        degradation_flag = 1

    # ======================================================
    # PERSISTENCE-BASED MALICIOUS LABELING
    # ======================================================

    key = (xapp_id, ue_id)

    degradation_history[key].append(degradation_flag)

    if len(degradation_history[key]) > PERSISTENCE_WINDOW:
        degradation_history[key].pop(0)

    rolling_degradation_count = sum(degradation_history[key])

    persistent_target_flag = int(
        rolling_degradation_count >= PERSISTENCE_THRESHOLD
        and xapp_id == ROGUE_XAPP
    )

    is_malicious = persistent_target_flag

    # ======================================================
    # RELATIONAL FEATURES
    # ======================================================

    message_frequency = len(degradation_history[key])
    repeat_target_count = len(degradation_history[key])
    unique_target_ratio = 1 / message_frequency if message_frequency > 0 else 0

    context_violation_score = int(
        (action != expected_action) and degradation_flag == 1
    )

    # ======================================================
    # STORE RECORD
    # ======================================================

    records.append([
        ts, xapp_id, ue_id, cell_id,
        action, expected_action, action_context_match,
        rsrp, sinr, cell_load,
        throughput, latency,
        performance_delta, degradation_flag,
        message_frequency, repeat_target_count,
        unique_target_ratio,
        rolling_degradation_count,
        persistent_target_flag,
        context_violation_score,
        is_malicious
    ])


# ==========================================================
# 6. DATAFRAME EXPORT
# ==========================================================

columns = [
    "timestamp", "xapp_id", "ue_id", "cell_id",
    "action_type", "expected_action", "action_context_match",
    "rsrp", "sinr", "cell_load",
    "throughput_mbps", "latency_ms",
    "performance_delta", "degradation_flag",
    "message_frequency", "repeat_target_count",
    "unique_target_ratio",
    "rolling_degradation_count",
    "persistent_target_flag",
    "context_violation_score",
    "is_malicious"
]

df = pd.DataFrame(records, columns=columns)

df.to_csv("dataset/context_violation_dataset.csv", index=False)

print("Dataset generated successfully.")
print("Total samples:", len(df))
print("Malicious samples:", df["is_malicious"].sum())
