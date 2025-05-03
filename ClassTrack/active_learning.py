import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from train_participation_model import ParticipationNet, features, scaler, X, y

# Configuration
THRESHOLD_PRECISION = 0.90
MIN_HIGH_QUALITY_MODELS = 10
METRICS_FILE = "metrics/weekly_report.csv"
LOG_DATA_PATH = "data/logs.csv"
STUDENT_DATA_PATH = "students.json"

# Check if model is ready for active learning
def model_ready_for_active_learning():
    if not os.path.exists(METRICS_FILE):
        return False
    df = pd.read_csv(METRICS_FILE)
    high_precision_models = df[df['precision'] >= THRESHOLD_PRECISION]
    return len(high_precision_models) >= MIN_HIGH_QUALITY_MODELS

if not model_ready_for_active_learning():
    print("Active learning not enabled yet — need at least 10 models with ≥90% precision.")
    exit()

print("Active Learning Activated — scanning for low confidence events...")

# Load student names
import json
if os.path.exists(STUDENT_DATA_PATH):
    with open(STUDENT_DATA_PATH, "r") as f:
        student_names = json.load(f)
else:
    student_names = {}

# Load model
model = ParticipationNet(input_size=len(features))
model.load_state_dict(torch.load("models/classifier.pt"))
model.eval()

X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    logits = model(X_tensor)
    probs = torch.softmax(logits, dim=1).numpy()

confidence = np.max(probs, axis=1)
predicted = np.argmax(probs, axis=1)

# Load raw logs
df_raw = pd.read_csv(LOG_DATA_PATH)
if 'name' not in df_raw.columns:
    df_raw['name'] = df_raw['student_id'].apply(lambda sid: student_names.get(str(sid), {}).get("name", ""))

# Attach predictions and confidence
df_raw['predicted_label'] = predicted
df_raw['confidence'] = confidence

# Sort by least confident
low_conf_df = df_raw.sort_values(by="confidence").head(10)

# Save to review file
os.makedirs("active_flags", exist_ok=True)
flag_file = f"active_flags/review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
low_conf_df.to_csv(flag_file, index=False)

print(f"Saved top 10 low-confidence predictions to {flag_file}")
print(low_conf_df[['student_id', 'name', 'confidence', 'predicted_label']])
