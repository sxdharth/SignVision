import cv2
import numpy as np
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

# The 30 Conversational Words
WORDS = [
    'hello', 'goodbye', 'thanks', 'please', 'sorry', 'how',
    'yes', 'no', 'good', 'bad', 'ready', 'wait',
    'what', 'who', 'where', 'why', 'when', 'help',
    'like', 'want', 'need', 'know', 'understand', 'work',
    'mother', 'father', 'friend', 'family',
    'now', 'later', 'name', 'me'
]

# We will collect 100 videos per word (Requested by user for higher accuracy)
SAMPLES_PER_WORD = 100
# Temporal Sequence Length
FRAMES_PER_VIDEO = 30
# Directory to immediately save extracted landmarks ready for ML training
DATA_DIR = os.path.join('Data', 'Custom_Processed')

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    # Request HD stream for clearer processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=========================================")
    print("SignVision S8 - Conversational Word Recorder")
    print("=========================================")
    
    for word in WORDS:
        word_dir = os.path.join(DATA_DIR, word)
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)
            
        print(f"\n--- Get ready to sign: {word.upper()} ---")
        time.sleep(2)
        
        for sample in range(SAMPLES_PER_WORD):
            # Check how many files we already have so we don't overwrite if script is restarted
            existing_files = os.listdir(word_dir)
            sample_num = len(existing_files)
            
            if sample_num >= SAMPLES_PER_WORD:
                print(f"[{word}] Already has {SAMPLES_PER_WORD} samples. Skipping.")
                break
                
            sequence_data = []
            
            # Wait for spacebar to start the 30-frame capture
            while True:
                ret, frame = cap.read()
                if not ret: continue
                cv2.putText(frame, f"Sign: {word.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                cv2.putText(frame, f"Video {sample_num+1}/{SAMPLES_PER_WORD}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, "Press SPACEBAR to start next recording...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("Data Collection", frame)
                
                key = cv2.waitKey(33) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    print("Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    extractor.close()
                    sys.exit(0)
                
                
            # Recording Loop (Extracting landmarks immediately on the fly)
            for frame_num in range(FRAMES_PER_VIDEO):
                ret, frame = cap.read()
                if not ret: continue
                
                # Extract landmarks
                landmarks = extractor.extract_landmarks(frame)
                sequence_data.append(landmarks)
                
                # Visual feedback
                cv2.putText(frame, f"RECORDING: {word.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv2.putText(frame, f"Frame {frame_num+1}/{FRAMES_PER_VIDEO}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)
                
                # Press 'q' to quit entirely
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    print("Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    extractor.close()
                    sys.exit(0)
                    
            # Save the temporal sequence array
            npy_path = os.path.join(word_dir, f"{sample_num}.npy")
            np.save(npy_path, np.array(sequence_data))
            print(f"Saved: {npy_path}")

    # Done
    print("\nAll 30 conversational words recorded successfully!")
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == '__main__':
    main()
