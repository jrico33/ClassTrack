
import os
import phase_manager

print("\n Starting Deep Learning Voice Recognition Pipeline")

# Check and print current system phase
phase = phase_manager.check_phase_transition()
phase_manager.print_phase()

# Step 1: Train Model
print("\n Training deep learning model...")
os.system("python train_participation_model.py")

# Step 2: Evaluate Model & log performance
print("\n Evaluating model performance...")
os.system("python evaluate_model.py")

# Step 3: If Phase ≥ 2, run active learning
if phase >= 2:
    print("\n Checking for low-confidence predictions (Active Learning)...")
    os.system("python active_learning.py")

print("\n Pipeline finished. Logs and models updated.")
