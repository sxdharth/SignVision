"""
Continuous Rolling Recorder for Video Call Model
================================================
Unlike the old recorder that captures exactly 30 pure-gesture frames,
this recorder simulates the LIVE WebRTC pipeline by continuously
recording 30-frame rolling windows that contain natural idle→gesture→idle
transitions — exactly what the model sees during a real video call.

Usage:
  1. Run this script.
  2. Select a sign from the list.
  3. When prompted, get ready and start performing the sign NATURALLY.
  4. The recorder captures multiple overlapping 30-frame windows
     during your gesture, each containing different mixtures of
     idle + gesture frames.
"""

import cv2
import os
import numpy as np
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../Data/Custom_Processed')
VIDEO_DIR = os.path.join(os.path.dirname(__file__), '../Dataset')
MAX_LENGTH = 30
SAMPLES_PER_TRIGGER = 8  # How many rolling windows to save per gesture performance

def get_existing_classes():
    """Discover all classes that already exist in Custom_Processed."""
    classes = []
    if os.path.exists(OUTPUT_DIR):
        for folder in sorted(os.listdir(OUTPUT_DIR)):
            if os.path.isdir(os.path.join(OUTPUT_DIR, folder)):
                classes.append(folder)
    return classes

def main():
    print("=" * 50)
    print("  CONTINUOUS ROLLING RECORDER")
    print("  (Realistic Idle→Gesture→Idle Training Data)")
    print("=" * 50)
    
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    classes = get_existing_classes()
    if not classes:
        print("No existing classes found. Enter a new class name.")
    else:
        print(f"\nExisting classes ({len(classes)}):")
        for i, cls in enumerate(classes):
            count = len([f for f in os.listdir(os.path.join(OUTPUT_DIR, cls)) if f.endswith('.npy')])
            print(f"  {i+1}. {cls} ({count} samples)")
    
    while True:
        print("\n--- Enter class name (or number), 'q' to quit ---")
        choice = input("> ").strip()
        
        if choice.lower() == 'q':
            break
        
        # Allow selecting by number
        if choice.isdigit() and 1 <= int(choice) <= len(classes):
            sign_name = classes[int(choice) - 1]
        else:
            sign_name = choice.lower()
        
        class_dir = os.path.join(OUTPUT_DIR, sign_name)
        os.makedirs(class_dir, exist_ok=True)
        existing = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        
        print(f"\nSelected: '{sign_name}' (has {existing} existing samples)")
        print("Instructions:")
        print("  1. Stay IDLE in front of the camera")
        print("  2. Press SPACE when ready to start your sign")
        print("  3. Perform the sign NATURALLY (take your time)")
        print("  4. The recorder will auto-capture rolling windows")
        print("  Press 'n' for next class, 'q' to quit\n")
        
        # Continuous rolling buffer
        rolling_buffer = []
        video_frames = []  # Raw video frames for saving clips
        
        # Create video output directory for this class
        video_class_dir = os.path.join(VIDEO_DIR, sign_name)
        os.makedirs(video_class_dir, exist_ok=True)
        
        recording_active = False
        capture_countdown = 0
        saved_this_round = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Always extract landmarks (simulates live pipeline)
            landmarks = extractor.extract_landmarks(frame)
            rolling_buffer.append(landmarks)
            
            # Keep only last MAX_LENGTH frames
            if len(rolling_buffer) > MAX_LENGTH:
                rolling_buffer.pop(0)
            
            if recording_active:
                # Also collect raw video frames for the video clip
                video_frames.append(frame.copy())
                
                # Save rolling windows at regular intervals during the gesture
                if len(rolling_buffer) == MAX_LENGTH:
                    # Save a snapshot every 3 frames to get diverse windows
                    capture_countdown += 1
                    if capture_countdown % 3 == 0 and saved_this_round < SAMPLES_PER_TRIGGER:
                        save_path = os.path.join(class_dir, f"rolling_{int(time.time() * 1000)}.npy")
                        np.save(save_path, np.array(rolling_buffer))
                        saved_this_round += 1
                        existing += 1
                
                cv2.putText(display, f"RECORDING '{sign_name}' - Saved: {saved_this_round}/{SAMPLES_PER_TRIGGER}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display, f"Total samples: {existing}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Auto-stop after collecting enough windows
                if saved_this_round >= SAMPLES_PER_TRIGGER:
                    recording_active = False
                    
                    # Save the video clip
                    if len(video_frames) > 0:
                        h, w = video_frames[0].shape[:2]
                        video_path = os.path.join(video_class_dir, f"{sign_name}_{int(time.time())}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                        for vf in video_frames:
                            out.write(vf)
                        out.release()
                        print(f"  -> Video saved to: {video_path}")
                    
                    video_frames = []
                    saved_this_round = 0
                    capture_countdown = 0
                    print(f"  -> Captured {SAMPLES_PER_TRIGGER} rolling windows! ({existing} total)")
                    print("  Press SPACE to record again, 'n' for next class")
            else:
                cv2.putText(display, f"Class: {sign_name} | Samples: {existing}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, "Press SPACE to start, 'n' for next, 'q' to quit", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            cv2.imshow('Continuous Rolling Recorder', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to start recording
                if not recording_active:
                    recording_active = True
                    saved_this_round = 0
                    capture_countdown = 0
                    video_frames = []  # Start fresh video clip
                    print("  Recording started! Perform your sign NOW...")
                    
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                print("Recorder closed.")
                return
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("\nRecorder closed. Run 'python tools/video_call_data_merger.py' to rebuild the dataset.")

if __name__ == "__main__":
    main()
