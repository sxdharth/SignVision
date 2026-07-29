import cv2
import numpy as np
import os
import sys
import time
import string

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

# The 26 Alphabetical Letters
LETTERS = list(string.ascii_lowercase)

# We will collect 200 static frames per letter to ensure absolute accuracy 
# from all slightly varying angles of the hand.
SAMPLES_PER_LETTER = 200

# Directory for static frames
DATA_DIR = os.path.join('Data', 'Spelling_Processed')

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=========================================")
    print("SignVision S8 - Alphabet Spelling Recorder")
    print("=========================================")
    print("Instructions: This will take 200 photos of each letter.")
    print("Please slowly rotate your hand and change your angle while it records")
    print("so the AI learns all perspectives of the finger positions.")
    print("=========================================")
    
    for letter in LETTERS:
        letter_dir = os.path.join(DATA_DIR, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
            
        existing_files = os.listdir(letter_dir)
        sample_num = len(existing_files)
        
        if sample_num >= SAMPLES_PER_LETTER:
            print(f"[{letter}] Already has {SAMPLES_PER_LETTER} samples. Skipping.")
            continue
            
        print(f"\n--- Get ready to hold the letter: {letter.upper()} ---")
        
        # Countdown
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if not ret: continue
            cv2.putText(frame, f"Letter: {letter.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            cv2.putText(frame, f"Starting in {i}...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.imshow("Spelling Training", frame)
            cv2.waitKey(1000)
            
        # Recording Loop
        while sample_num < SAMPLES_PER_LETTER:
            ret, frame = cap.read()
            if not ret: continue
            
            # Extract landmarks
            landmarks = extractor.extract_landmarks(frame)
            
            # If no hands are detected, skip saving this frame
            if np.all(landmarks[99:162] == 0) and np.all(landmarks[162:] == 0):
                cv2.putText(frame, "NO HAND DETECTED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # Save just the single flattened array (shape: (225,))
                npy_path = os.path.join(letter_dir, f"{sample_num}.npy")
                np.save(npy_path, landmarks)
                sample_num += 1
            
            # Visual feedback
            cv2.putText(frame, f"RECORDING LETTER: {letter.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.putText(frame, f"Photos taken: {sample_num}/{SAMPLES_PER_LETTER}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "(Slowly rotate your hand/wrist)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Spelling Training", frame)
            
            # Press 'q' to quit entirely
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                sys.exit(0)
                
        print(f"Finished recording {letter.upper()}.")

    # Done
    print("\nAll 26 Alphabet letters recorded successfully!")
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == '__main__':
    main()
