import streamlit as st
import os
import json
import pandas as pd
import subprocess
import sys
import time
import glob
import shutil

RUN_FLAG = "voice_detect_running.txt"
TAG_FILE = "tag_state.json"
LOG_FILE = "data/logs.csv"
VOICES_FILE = "voices.json"

st.set_page_config(layout="wide")
st.title("ClassTrack")


def save_tag_state(active, student_id):
    with open(TAG_FILE, "w") as f:
        json.dump({"active": active, "student_id": student_id}, f)


def archive_old_log():
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        os.makedirs("data/old_logs", exist_ok=True)
        existing_logs = glob.glob("data/old_logs/logs*.csv")
        log_numbers = [int(os.path.basename(f).replace("logs", "").replace(".csv", "")) for f in existing_logs]
        next_number = max(log_numbers, default=0) + 1
        shutil.move(LOG_FILE, f"data/old_logs/logs{next_number}.csv")


def stop_voice_detect():
    if os.path.exists(RUN_FLAG):
        os.remove(RUN_FLAG)
    st.success("Class session stopped.")


def start_voice_detect():
    archive_old_log()
    with open(LOG_FILE, "w") as f:
        f.write(
            "start_time,end_time,student_id,name,avg_volume,duration_seconds,is_overlapping,was_teacher_before,overlapped_with_teacher,unique_speaker_count,tag\n")
    subprocess.Popen([sys.executable, "voice_detect.py"])
    time.sleep(1)
    st.success("Class session started.")


# --- Load Roster and Student Names ---
roster_df = pd.read_csv("roster.csv", header=None)
student_ids = [sid for sid in roster_df[0].astype(str)]

with open(VOICES_FILE, "r") as f:
    voices_db = json.load(f)

students = [(sid, voices_db[sid]["name"]) for sid in student_ids if not sid.startswith("T")]

# --- Session Control ---
st.markdown("##")

col1, col2 = st.columns([1, 4])
with col1:
    if not os.path.exists(RUN_FLAG):
        if st.button("▶️ Start Class Session", use_container_width=True):
            start_voice_detect()
            st.rerun()
    else:
        if st.button("🛑 Stop Class Session", use_container_width=True):
            stop_voice_detect()
            st.rerun()

# --- Student Tagging ---
st.subheader("Tag a Participating Student")

cols = st.columns(6)
for i, (sid, name) in enumerate(students):
    with cols[i % 6]:
        st.button(
            label=name,
            key=f"student_{sid}",
            use_container_width=True,
            help=f"Tag {name} as participating",
            on_click=save_tag_state,
            kwargs={"active": True, "student_id": sid}
        )

st.markdown("##")
st.divider()

if st.button("❌ Clear Tag", use_container_width=True):
    save_tag_state(False, None)
    st.info("No active student tag.")

# --- Class Results ---
if os.path.exists(LOG_FILE) and not os.path.exists(RUN_FLAG):
    st.subheader("Class Session Results")
    try:
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            df = df[df["student_id"].apply(lambda x: not str(x).startswith("T"))]
            df["start_time"] = pd.to_datetime(df["start_time"], unit="s")

            # ✅ GROUP BY student_id and name together
            summary = df.groupby(["student_id", "name"]).agg(
                total_events=("start_time", "count"),
                avg_volume=("avg_volume", "mean"),
                participations=("tag", lambda x: (x == "participation").sum()),
                disruptions=("tag", lambda x: (x == "disruption").sum()),
            ).reset_index()

            summary = summary.sort_values("total_events", ascending=False)

            st.dataframe(summary, use_container_width=True)

            st.download_button(
                label="Download Current Class Log",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="class_log.csv",
                use_container_width=True
            )
        else:
            st.info("No events recorded yet.")
    except Exception as e:
        st.error(f"Error loading results: {e}")
else:
    if not os.path.exists(RUN_FLAG):
        st.info("No completed session to show yet.")


