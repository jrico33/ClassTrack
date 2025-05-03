import os
import numpy as np
import json
from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()

RECORDINGS_DIR = "recordings"
STUDENT_DB_PATH = "students.json"
ROSTER_PATH = "roster.csv"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load student database (with sample paths)
if not os.path.exists(STUDENT_DB_PATH):
    raise FileNotFoundError("'students.json' not found. Run voice_collector.py first.")

with open(STUDENT_DB_PATH, "r") as f:
    student_db = json.load(f)

# Load class roster
if not os.path.exists(ROSTER_PATH):
    raise FileNotFoundError("'roster.csv' not found. Please upload the teacher's class list.")

with open(ROSTER_PATH, "r") as f:
    roster_ids = [line.strip() for line in f if line.strip()]

print(f"Loaded roster with {len(roster_ids)} students.")

embeddings = []
names = []

for student_id in roster_ids:
    if student_id not in student_db:
        print(f"No recordings found for student {student_id}. Skipping.")
        continue

    sample_paths = student_db[student_id]["samples"]
    speaker_embeddings = []

    for path in sample_paths:
        if not os.path.exists(path):
            print(f"Missing recording: {path}")
            continue
        wav = preprocess_wav(path)
        emb = encoder.embed_utterance(wav)
        speaker_embeddings.append(emb)

    if len(speaker_embeddings) < 2:
        print(f"Not enough valid recordings for {student_id}. Skipping.")
        continue

    avg_embedding = np.mean(speaker_embeddings, axis=0)
    embeddings.append(avg_embedding)
    names.append(student_id)

# Save usable data
np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), np.array(embeddings))
np.save(os.path.join(OUTPUT_DIR, "names.npy"), np.array(names))

print(f"\nEmbeddings generated and saved for {len(embeddings)} students.")