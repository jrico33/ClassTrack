import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

#Deep MLP model
class ParticipationNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Participation or Disruption
        )

    def forward(self, x):
        return self.net(x)


#Load data
LOG_PATH = "data/logs.csv"
assert os.path.exists(LOG_PATH), "Log file not found!"
df = pd.read_csv(LOG_PATH)

#Optional: use name field for later NLP upgrades
if 'name' not in df.columns:
    df['name'] = ""

#Feature columns we care about
features = [
    'avg_volume', 'duration_seconds',
    'is_overlapping', 'was_teacher_before',
    'overlapped_with_teacher', 'unique_speaker_count'
]

# Encode features
X = df[features].fillna(0).values
y = df['tag'].map({"participation": 0, "disruption": 1}).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

#Model setup
model = ParticipationNet(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training loop
print("\nTraining model...")
for epoch in range(50):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")

#Evaluation
print("\nEvaluating model...")
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, axis=1)

print("\nAccuracy:", (predicted_labels == y_test).float().mean().item())
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predicted_labels))
print("\nClassification Report:\n", classification_report(y_test, predicted_labels, target_names=["participation", "disruption"]))

#Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/classifier.pt")
print("\nModel saved to models/classifier.pt")