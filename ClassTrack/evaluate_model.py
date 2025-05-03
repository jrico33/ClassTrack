import os
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from train_participation_model import ParticipationNet, features, scaler, X_test, y_test
from datetime import datetime

# Reload model
MODEL_PATH = "models/classifier.pt"
assert os.path.exists(MODEL_PATH), "No trained model found!"

model = ParticipationNet(input_size=len(features))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, axis=1).numpy()

# Scores
precision = precision_score(y_test, predicted_labels, zero_division=0)
recall = recall_score(y_test, predicted_labels, zero_division=0)
f1 = f1_score(y_test, predicted_labels, zero_division=0)
acc = accuracy_score(y_test, predicted_labels)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report = {
    "timestamp": now,
    "accuracy": round(acc, 3),
    "precision": round(precision, 3),
    "recall": round(recall, 3),
    "f1": round(f1, 3)
}

# Append readiness flag
metrics_path = "metrics/weekly_report.csv"
os.makedirs("metrics", exist_ok=True)

if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)
    high_precision = df[df['precision'] >= 0.90]
    report['active_learning_ready'] = len(high_precision) + 1 >= 10
else:
    report['active_learning_ready'] = False

# Save new row
log_exists = os.path.exists(metrics_path)
df_new = pd.DataFrame([report])
df_new.to_csv(metrics_path, mode='a', header=not log_exists, index=False)

print("\nEvaluation Logged:")
print(df_new)

# Optional: print full classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels, target_names=["participation", "disruption"]))

# Notify phase transition
if report['active_learning_ready']:
    print("\nModel has reached Phase 2 (Shadow Prediction + Active Learning enabled)")