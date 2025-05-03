import os
import json
import sounddevice as sd
from scipy.io.wavfile import write


sd.default.device = (1, None)
RECORDINGS_DIR = "recordings"
JSON_PATH = "voices.json"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

sentences = [
    "Hello, my name is {student_name}, and I am part of IE university.",
    "The quick brown fox jumps over the lazy dog.",
    "She had your dark suit in greasy wash water all year.",
    "We enjoyed watching the movie together the other day."
]

def load_student_db():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    return {}

def save_student_db(db):
    with open(JSON_PATH, "w") as f:
        json.dump(db, f, indent=2)

def record_audio(filename, duration=5, fs=16000):
    print(f"Recording for {duration} seconds. Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"Saved: {filename}")

def is_teacher_id(student_id):
    return str(student_id).strip().upper().startswith("T")

def add_student(student_db):
    while True:
        student_id = input("\nWrite down your Student/Teacher ID (or type 'done' to finish): ").strip()
        if student_id.lower() == "done":
            break

        if student_id in student_db:
            print(f"Student/Teacher ID '{student_id}' already exists.")
            choice = input("Do you want to [U]pdate recordings, [C]hange ID, or [S]kip? ").strip().lower()
            if choice == "u":
                print(f"Updating recordings for {student_id}.")
            elif choice == "c":
                continue
            else:
                print("Skipping...")
                continue
        else:
            print(f"Registering new ID: {student_id}")

        if is_teacher_id(student_id):
            student_name = input("Enter the full name of the TEACHER: ").strip()
        else:
            student_name = input("Enter the full name of the STUDENT: ").strip()

        print(f"Name saved as: {student_name}")

        paths = []
        for i, template in enumerate(sentences):
            phrase = template.format(student_name=student_name)
            print(f"\nPlease say:\n\"{phrase}\"")
            filename = os.path.join(RECORDINGS_DIR, f"{student_id}_s{i+1}.wav")
            record_audio(filename)
            paths.append(filename)

        student_db[student_id] = {
            "name": student_name,
            "samples": paths
        }
        save_student_db(student_db)

        print(f"ID '{student_id}' ({student_name}) recorded and saved.")

if __name__ == "__main__":
    print("Voice Collector for Students and Teachers")
    student_db = load_student_db()
    add_student(student_db)
    print("Done. All entries saved to students.json.")