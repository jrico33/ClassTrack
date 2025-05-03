import os
import json
import pandas as pd
from datetime import datetime

METRICS_PATH = "metrics/weekly_report.csv"
STATE_PATH = "system_state.json"

PHASES = {
    1: "Manual Tagging",
    2: "Shadow Prediction + Active Learning",
    3: "Autonomous (≥95%)",
    4: "Fully Independent AI (≥99%)"
}

def get_current_phase():
    if not os.path.exists(STATE_PATH):
        return 1  # Default to Phase 1
    with open(STATE_PATH, "r") as f:
        return json.load(f).get("phase", 1)

def save_phase(phase):
    with open(STATE_PATH, "w") as f:
        json.dump({"phase": phase, "last_updated": datetime.now().isoformat()}, f, indent=2)

def check_consecutive_days(df, threshold, days=7):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_success = df.groupby('date')['precision'].apply(lambda x: (x >= threshold).all())
    streak = 0
    for success in daily_success[::-1]:
        if success:
            streak += 1
            if streak >= days:
                return True
        else:
            break
    return False

def check_phase_transition():
    if not os.path.exists(METRICS_PATH):
        return get_current_phase()

    df = pd.read_csv(METRICS_PATH)
    phase = get_current_phase()

    # Phase 2 Trigger: 10+ models with ≥90% precision (any time)
    if phase < 2:
        high_precision = df[df['precision'] >= 0.90]
        if len(high_precision) >= 10:
            save_phase(2)
            print("Transitioned to Phase 2: Shadow Prediction + Active Learning Enabled")
            return 2

    # Phase 3 Trigger: 7 consecutive days all models ≥95%
    if phase < 3 and check_consecutive_days(df, threshold=0.95, days=7):
        save_phase(3)
        print("Transitioned to Phase 3: Autonomous Mode (≥95%)")
        return 3

    # Phase 4 Trigger: 7 consecutive days all models ≥99%
    if phase < 4 and check_consecutive_days(df, threshold=0.99, days=7):
        save_phase(4)
        print("Transitioned to Phase 4: Fully Independent AI (≥99%)")
        return 4

    return phase

def print_phase():
    current = get_current_phase()
    print(f"Current Phase: {current} - {PHASES[current]}")

if __name__ == "__main__":
    phase = check_phase_transition()
    print_phase()