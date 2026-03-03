import cv2
import os
import numpy as np
import time
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

OUTPUT_DIR = 'Data/Custom_Processed'

def record_signs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    while True:
        sign_name = input("Enter the sign name you want to record (or 'q' to quit): ").strip()
        if sign_name.lower() == 'q':
            break
            
        class_dir = os.path.join(OUTPUT_DIR, sign_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        print(f"Recording for class: '{sign_name}'.")
        print("Controls:")
        print("  'r': Start/Stop recording a sequence")
        print("  'n': Enter new class name")
        print("  'q': Quit application")
        
        is_recording = False
        frames = []
        sequence_count = len(os.listdir(class_dir))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            if is_recording:
                cv2.putText(display_frame, "RECORDING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                landmarks = extractor.extract_landmarks(frame)
                frames.append(landmarks)
            else:
                cv2.putText(display_frame, f"Class: {sign_name} | Count: {sequence_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'r' to record, 'n' for new class", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('Custom Recorder', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                if not is_recording:
                    is_recording = True
                    frames = []
                    print("Started recording...")
                else:
                    is_recording = False
                    print("Stopped recording.")
                    if len(frames) > 10: # Minimum frames threshold
                        save_path = os.path.join(class_dir, f"{int(time.time())}.npy")
                        np.save(save_path, np.array(frames))
                        sequence_count += 1
                        print(f"Saved sequence {sequence_count} to {save_path}")
                    else:
                        print("Sequence too short, discarded.")
            
            elif key == ord('n'):
                break
            
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                return

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == "__main__":
    record_signs()
