"""
Diagnostic: Compare landmark extraction between training pipeline and live pipeline.
This simulates both paths and measures the difference.

Training path:  cv2.VideoCapture -> raw BGR frame -> extract_landmarks
Live path:      cv2.VideoCapture -> resize 640x480 -> JPEG 80% -> decode -> extract_landmarks

Run: .venv\Scripts\python.exe diagnose_pipeline.py
"""
import os, sys, json
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, 'src'))

from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

# Get native resolution
native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

results = []

print(f"Camera native resolution: {native_w}x{native_h}")
print(f"Capturing 10 frames for comparison...\n")

# Skip first few frames for camera warmup
for _ in range(10):
    cap.read()

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    # === PATH A: Training pipeline (raw frame) ===
    landmarks_raw = extractor.extract_landmarks(frame)

    # === PATH B: Live pipeline (resize + JPEG compress + decode) ===
    resized = cv2.resize(frame, (640, 480))
    _, jpeg_buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
    decoded = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
    landmarks_live = extractor.extract_landmarks(decoded)

    # === PATH C: Live pipeline with higher quality JPEG ===
    _, jpeg_hq = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    decoded_hq = cv2.imdecode(jpeg_hq, cv2.IMREAD_COLOR)
    landmarks_hq = extractor.extract_landmarks(decoded_hq)

    # === PATH D: Live pipeline with PNG (lossless) ===
    _, png_buf = cv2.imencode('.png', resized)
    decoded_png = cv2.imdecode(png_buf, cv2.IMREAD_COLOR)
    landmarks_png = extractor.extract_landmarks(decoded_png)

    # Compare
    diff_live = np.mean(np.abs(landmarks_raw - landmarks_live))
    diff_hq = np.mean(np.abs(landmarks_raw - landmarks_hq))
    diff_png = np.mean(np.abs(landmarks_raw - landmarks_png))

    # Also check hand presence
    hands_raw = np.any(landmarks_raw[99:] != 0)
    hands_live = np.any(landmarks_live[99:] != 0)

    results.append({
        'diff_jpeg80': diff_live,
        'diff_jpeg95': diff_hq,
        'diff_png': diff_png,
        'hands_raw': hands_raw,
        'hands_live': hands_live,
    })

    print(f"Frame {i+1}: JPEG80 diff={diff_live:.6f}  JPEG95 diff={diff_hq:.6f}  PNG diff={diff_png:.6f}  hands_raw={hands_raw} hands_live={hands_live}")

cap.release()
extractor.close()

# Write summary
with open(os.path.join(ROOT, 'pipeline_diagnosis.py'), 'w', encoding='utf-8') as f:
    avg_80 = np.mean([r['diff_jpeg80'] for r in results])
    avg_95 = np.mean([r['diff_jpeg95'] for r in results])
    avg_png = np.mean([r['diff_png'] for r in results])

    f.write(f"# Pipeline Diagnosis Results\n")
    f.write(f"# Camera native resolution: {native_w}x{native_h}\n")
    f.write(f"# Live pipeline: resized to 640x480 then compressed\n")
    f.write(f"#\n")
    f.write(f"# Average Landmark Difference (Mean Absolute Error):\n")
    f.write(f"#   JPEG 80% quality: {avg_80:.6f}\n")
    f.write(f"#   JPEG 95% quality: {avg_95:.6f}\n")
    f.write(f"#   PNG (lossless):   {avg_png:.6f}\n")
    f.write(f"#\n")
    f.write(f"# Per-frame details:\n")
    for i, r in enumerate(results):
        f.write(f"#   Frame {i+1}: jpeg80={r['diff_jpeg80']:.6f} jpeg95={r['diff_jpeg95']:.6f} png={r['diff_png']:.6f} hands_raw={r['hands_raw']} hands_live={r['hands_live']}\n")
    f.write(f"#\n")
    if avg_png > 0.001:
        f.write(f"# CONCLUSION: Significant landmark differences even with lossless PNG.\n")
        f.write(f"#   This means RESOLUTION CHANGE (not compression) is the main issue.\n")
        f.write(f"#   Fix: Send frames at native camera resolution, not 640x480.\n")
    elif avg_80 > 0.01 and avg_95 < avg_80 * 0.5:
        f.write(f"# CONCLUSION: JPEG compression at 80% is the main issue.\n")
        f.write(f"#   Fix: Increase JPEG quality to 95% or use PNG.\n")
    elif avg_80 < 0.001:
        f.write(f"# CONCLUSION: Pipeline differences are minimal.\n")
        f.write(f"#   Issue may be temporal (frame timing) or model overfitting.\n")
    else:
        f.write(f"# CONCLUSION: Mixed factors. Consider both resolution and compression.\n")

print(f"\nSummary written to pipeline_diagnosis.py")
print(f"Average diff JPEG80: {avg_80:.6f}")
print(f"Average diff JPEG95: {avg_95:.6f}")
print(f"Average diff PNG:    {avg_png:.6f}")
