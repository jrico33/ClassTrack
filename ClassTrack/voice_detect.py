import os
import time
import numpy as np
import pandas as pd
import json
import traceback
import sounddevice as sd
from resemblyzer import preprocess_wav, VoiceEncoder
from scipy.io.wavfile import write

# Force correct microphone
sd.default.device = (2, None)

# Configuration
FS = 16000
RECORD_SECONDS = 3
RUN_FLAG = "voice_detect_running.txt"

# Normalize function (⚡ Boost weak recordings smartly)
def normalize_audio(wav, target_level=0.1):
    peak = np.max(np.abs(wav))
    if peak == 0:
        return wav
    gain = target_level / peak
    return wav * gain

def is_teacher_id(student_id):
    return str(student_id).strip().upper().startswith("T")

def main():
    try:
        print("voice_detect.py has started")

        os.makedirs("data", exist_ok=True)
        log_path = "data/logs.csv"

        # Mark as running
        with open(RUN_FLAG, "w") as f:
            f.write("running")

        encoder = VoiceEncoder()

        # Load student database
        roster_ids = set(pd.read_csv("roster.csv", header=None)[0].astype(str))
        with open("voices.json", "r") as f:
            full_db = json.load(f)

        student_db = {k: v for k, v in full_db.items() if k in roster_ids}

        embeddings = {}
        speaker_types = {}

        for student_id, data in student_db.items():
            speaker_types[student_id] = "teacher" if is_teacher_id(student_id) else "student"
            wavs = [preprocess_wav(p) for p in data['samples']]
            embeddings[student_id] = VoiceEncoder().embed_utterance(np.concatenate(wavs))

        with open(log_path, "w") as f:
            f.write("start_time,end_time,student_id,name,avg_volume,duration_seconds,is_overlapping,was_teacher_before,overlapped_with_teacher,unique_speaker_count,tag\n")

        last_speaker = None
        current_tag = None

        print("Recording started. Ctrl+C to stop.\n")

        while os.path.exists(RUN_FLAG):
            start = time.time()

            audio = sd.rec(int(RECORD_SECONDS * FS), samplerate=FS, channels=1)
            sd.wait()
            wav = np.squeeze(audio)
            avg_vol_before = np.abs(wav).mean()

            # Apply normalization
            wav = normalize_audio(wav, target_level=0.1)
            avg_vol = np.abs(wav).mean()

            if avg_vol_before < 0.001:
                print(f"Low volume detected ({avg_vol_before:.5f}), skipping...\n")
                continue

            write("temp_test.wav", FS, wav)
            wav_pp = preprocess_wav("temp_test.wav")
            embed = encoder.embed_utterance(wav_pp)

            scores = {k: np.dot(embed, v) for k, v in embeddings.items()}
            best_id, best_score = max(scores.items(), key=lambda x: x[1])
            best_type = speaker_types.get(best_id, "unknown")

            end = time.time()
            duration = round(end - start, 2)

            is_overlap = last_speaker is not None and last_speaker != best_id
            was_teacher = bool(last_speaker) and speaker_types.get(last_speaker) == "teacher"
            overlapped_teacher = is_overlap and was_teacher

            speaker_names = [k for k, v in scores.items() if v > 0.75]
            speaker_count = len(speaker_names)

            # Load current tag
            current_tag = None
            if os.path.exists("tag_state.json"):
                with open("tag_state.json", "r") as f:
                    tag_data = json.load(f)
                    if tag_data.get("active"):
                        current_tag = "participation"

            tag = current_tag or "disruption"

            print(f"Detected: {best_id} ({best_type}) Tag: {tag} Overlap: {is_overlap}")

            if best_type == "teacher":
                last_speaker = best_id
                continue

            with open(log_path, "a") as f:
                f.write(f"{start},{end},{best_id},{student_db[best_id].get('name','')},{avg_vol_before},{duration},{int(is_overlap)},{int(was_teacher)},{int(overlapped_teacher)},{speaker_count},{tag}\n")

            last_speaker = best_id

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        if os.path.exists(RUN_FLAG):
            os.remove(RUN_FLAG)
        print("\nSession ended. Logs saved.")

if __name__ == "__main__":
    main()
