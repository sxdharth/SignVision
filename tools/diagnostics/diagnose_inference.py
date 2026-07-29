"""
Diagnose train/inference mismatch.
Compare landmark stats from training data vs live webcam.
Run this: python diagnose_inference.py
"""
import os, sys, json, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, 'src'))
from src.feature_extractor import FeatureExtractor
from tensorflow.keras.models import load_model

DATA_DIR  = os.path.join(ROOT, 'Data', 'Video_Call_Raw')
MODEL_PATH = os.path.join(ROOT, 'Models', 'video_call_model.h5')
CLASSES   = json.load(open(os.path.join(ROOT, 'Data', 'video_call_classes.json')))
INV       = {v: k for k, v in CLASSES.items()}
MAX_LEN   = 30

print("Loading model...")
model = load_model(MODEL_PATH)
extractor = FeatureExtractor()

# ── 1. Check one training sample directly ─────────────────────────────────────
print("\n=== TRAINING DATA SAMPLE PREDICTION ===")
for sign in list(CLASSES.keys())[:3]:
    class_dir = os.path.join(DATA_DIR, sign)
    if not os.path.isdir(class_dir): continue
    files = [f for f in os.listdir(class_dir) if f.endswith('.npy')][:1]
    if not files: continue
    seq = np.load(os.path.join(class_dir, files[0]))  # shape (30, feat)
    res = model.predict(np.expand_dims(seq, 0), verbose=0)[0]
    top = np.argsort(res)[::-1][:3]
    print(f"  [{sign}] → top pred: {INV[top[0]]} ({res[top[0]]:.2%})  "
          f"2nd: {INV[top[1]]} ({res[top[1]]:.2%})  "
          f"3rd: {INV[top[2]]} ({res[top[2]]:.2%})")

# ── 2. Live webcam capture and predict ────────────────────────────────────────
print("\n=== LIVE WEBCAM TEST ===")
print("Capturing 30 frames from webcam...")
cap = cv2.VideoCapture(0)
frames = []
start = time.time()
while len(frames) < MAX_LEN:
    ret, frame = cap.read()
    if not ret: continue
    cv2.imshow("Capturing... press Q to quit", frame)
    cv2.waitKey(1)
    lm = extractor.extract_landmarks(frame)
    frames.append(lm)
cap.release()
cv2.destroyAllWindows()
elapsed = time.time() - start
print(f"Captured {len(frames)} frames in {elapsed:.1f}s  => effective FPS: {len(frames)/elapsed:.1f}")

seq = np.array(frames, dtype='float32')
print(f"Sequence shape: {seq.shape}  — non-zero: {np.count_nonzero(seq.sum(axis=1))}/30 frames have landmarks")
# Show landmark stats
print(f"  mean absolute value: {np.abs(seq).mean():.4f}")
print(f"  max value:           {seq.max():.4f}")
print(f"  min value:           {seq.min():.4f}")

res = model.predict(np.expand_dims(seq, 0), verbose=0)[0]
top = np.argsort(res)[::-1][:4]
print("\nLive prediction (no sign / idle):")
for i in top[:4]:
    print(f"  {INV[i]:<10} {res[i]:.2%}")

# ── 3. Landmark stats from training data ─────────────────────────────────────
print("\n=== TRAINING DATA LANDMARK STATS ===")
X = np.load(os.path.join(ROOT, 'Data', 'X_video_call.npy'))
print(f"  mean absolute value: {np.abs(X).mean():.4f}")
print(f"  max value:           {X.max():.4f}")
print(f"  min value:           {X.min():.4f}")
print(f"  shape:               {X.shape}")
print("\nDone.")
extractor.close()
