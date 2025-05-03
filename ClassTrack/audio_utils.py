from datetime import datetime
import csv
import os
import sounddevice as sd
# Force correct microphone
sd.default.device = (2, None)

def record_audio(filename, duration=5, fs=16000):
    print(f"Recording for {duration} seconds. Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print(f"Saved: {filename}")

def log_speaker_event(student_id, volume, pitch, overlaps_teacher=False, hand_raised=False, teacher_feedback=None):
    os.makedirs("data", exist_ok=True)
    with open("data/logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            student_id,
            round(volume, 4),
            round(pitch, 4),
            overlaps_teacher,
            hand_raised,
            teacher_feedback or "unknown"
        ])


if __name__ == "__main__":
    import sounddevice as sd
    from scipy.io.wavfile import write
    import numpy as np

    fs = 16000
    duration = 3

    print("Recording for 3 seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    print("Audio shape:", audio.shape)
    print("Max volume:", np.abs(audio).max())

    write("test_mic.wav", fs, audio)
    print("Saved to test_mic.wav")
