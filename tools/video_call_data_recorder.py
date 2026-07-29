import cv2
import os
import numpy as np
import time
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

# ============================================================
# IMPORTANT: These must match video_call_classes.json exactly.
# You will sign each word in front of your OWN webcam.
# This data will be used to retrain the model for YOUR style.
# ============================================================
VOCABULARY = [
    'goodbye', 'hello', 'how', 'please', 'sorry', 'thanks', 'yes'
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../Data/Video_Call_Raw')
TARGET_SAMPLES = 200  # 200 samples per word for solid generalization

def record_signs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Initializing Feature Extractor (MediaPipe)...")
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    print("\n--- SIGNVISION VIDEO CALL DATA RECORDER ---")
    print(f"Goal: Record {TARGET_SAMPLES} samples for each of the {len(VOCABULARY)} words.")
    print("Press 's' to START recording the current class.")
    print("Press 'n' to SKIP to the next class.")
    print("Press 'q' at any time to QUIT the application.\n")
    
    for sign_name in VOCABULARY:
        class_dir = os.path.join(OUTPUT_DIR, sign_name)
        os.makedirs(class_dir, exist_ok=True)
        
        sequence_count = len(os.listdir(class_dir))
        if sequence_count >= TARGET_SAMPLES:
            print(f"[{sign_name}] already has {sequence_count}/{TARGET_SAMPLES} samples. Skipping.")
            continue
            
        print(f"\n=============================================")
        print(f" NEXT SIGN: '{sign_name.upper()}'")
        print(f" Samples needed: {TARGET_SAMPLES - sequence_count}")
        print(f"=============================================")
        print("Get ready. Press 's' to start recording this class, or 'n' to skip.")
        
        # Wait for user to start or skip this class
        skip_class = False
        empty_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                empty_frames += 1
                if empty_frames > 50:
                    print("\n[CRITICAL ERROR] Camera could not be read! Ensure no other apps (Zoom, Web Browser) are using the camera and restart.")
                    return cleanup(cap, extractor)
                continue
            
            empty_frames = 0
            display = frame.copy()
            cv2.putText(display, f"NEXT: {sign_name.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display, "Press 's' to Start | 'n' to Skip | 'q' to Quit", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow('Video Call Recorder', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('n'):
                skip_class = True
                break
            elif key == ord('q'): return cleanup(cap, extractor)
            
        if skip_class:
            continue
            
        # Record samples until target is reached
        while sequence_count < TARGET_SAMPLES:
            print(f"\nRecording sample {sequence_count + 1}/{TARGET_SAMPLES} for '{sign_name}'...")
            
            # 2 second countdown before recording starts
            for i in range(2, 0, -1):
                ret, frame = cap.read()
                display = frame.copy()
                cv2.putText(display, f"Get Ready... {i}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow('Video Call Recorder', display)
                cv2.waitKey(1000)
                
            frames = []
            start_time = time.time()
            
            # Record exactly 30 frames (approx 2 seconds)
            while len(frames) < 30:
                ret, frame = cap.read()
                if not ret: break
                
                display = frame.copy()
                landmarks = extractor.extract_landmarks(frame)
                frames.append(landmarks)
                
                # Visual feedback
                progress = int((len(frames) / 30.0) * 100)
                cv2.putText(display, f"Recording '{sign_name}': {progress}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Video Call Recorder', display)
                
                # Check for emergency quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return cleanup(cap, extractor)
            
            if len(frames) == 30:
                save_path = os.path.join(class_dir, f"{int(time.time() * 1000)}.npy")
                np.save(save_path, np.array(frames))
                sequence_count += 1
                print(f" -> Saved! ({sequence_count}/{TARGET_SAMPLES} complete)")
            else:
                print(" -> Error: Sequence interrupted.")
                
            # Brief pause between samples
            time.sleep(0.5)

    print("\n=============================================")
    print("ALL TARGET SAMPLES COLLECTED FOR ALL WORDS!")
    print("=============================================")
    cleanup(cap, extractor)

def cleanup(cap, extractor):
    print("Closing camera and extractor...")
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == "__main__":
    record_signs()
