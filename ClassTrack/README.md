# ClassTrack

ClassTrack is a smart classroom monitoring system that uses voice recognition and machine learning to track student participation and disruptions in real time.

---

## Project Structure

```
pythonProject/
├── data/
│   ├── old_logs/
│   └── logs.csv
├── metrics/
│   └── weekly_report.csv
├── models/
├── recordings/
│
├── active_learning.py
├── app_streamlit.py
├── audio_utils.py
├── evaluate_model.py
├── generate_embeddings_from_roster.py
├── main.py
├── phase_manager.py
├── README.md
├── requirements.txt
├── roster.csv
├── tag_state.json
├── train_participation_model.py
├── voice_collector.ipynb
├── voice_detect.py
└──voices.json
```
---

## Setup

1Create a virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
---

## ️ How to Use

1. Record student voices:
   ```bash
   python voice_collector.py
   ```

2. Generate voice embeddings:
   ```bash
   python generate_embeddings_from_roster.py
   ```

3. Launch class dashboard:
   ```bash
   streamlit run app_streamlit.py
   ```

4. Train model on collected data:
   ```bash
   python train_participation_model.py
   ```

---

##  Evaluation Phases

- **Phase 1**: Manual tagging
- **Phase 2**: Shadow prediction + feedback
- **Phase 3**: Model ≥95% precision for 7 days
- **Phase 4**: Model ≥99% precision for 7 days → Fully autonomous

---