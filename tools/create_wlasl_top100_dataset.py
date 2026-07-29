import os
import json
import cv2
import numpy as np
import sys

# Ensure we can import from src
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.feature_extractor import FeatureExtractor
from tensorflow.keras.utils import to_categorical

WLASL_JSON_PATH = os.path.join(ROOT_DIR, 'wlasl comparison', 'WLASL_v0.3.json')
VIDEOS_DIR = os.path.join(ROOT_DIR, 'wlasl comparison', 'videos')
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

OUT_X = os.path.join(DATA_DIR, "X_wlasl100.npy")
OUT_Y = os.path.join(DATA_DIR, "y_wlasl100.npy")
OUT_CLASSES = os.path.join(DATA_DIR, "wlasl100_classes.json")

MAX_SEQ_LENGTH = 30

def process_sequence(seq):
    if len(seq) == 0:
        return None
        
    feature_size = seq.shape[1] if len(seq.shape) > 1 else seq[0].shape[0]
    
    if len(seq) > MAX_SEQ_LENGTH:
        indices = np.linspace(0, len(seq) - 1, MAX_SEQ_LENGTH, dtype=int)
        base_seq = seq[indices]
    elif len(seq) < MAX_SEQ_LENGTH:
        pad_amount = MAX_SEQ_LENGTH - len(seq)
        padding = np.zeros((pad_amount, feature_size))
        base_seq = np.vstack((seq, padding))
    else:
        base_seq = seq
        
    return base_seq

def main():
    print(f"Loading WLASL JSON...")
    with open(WLASL_JSON_PATH, 'r') as f:
        content = json.load(f)
        
    # Sort descending by instance count
    content.sort(key=lambda x: len(x['instances']), reverse=True)
    top_100_entries = content[:100]
    
    class_map = {entry['gloss']: idx for idx, entry in enumerate(top_100_entries)}
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    with open(OUT_CLASSES, 'w') as f:
        json.dump(class_map, f, indent=4)
        
    extractor = FeatureExtractor()
    
    X_data = []
    y_data = []
    
    for entry in top_100_entries:
        gloss = entry['gloss']
        label_idx = class_map[gloss]
        print(f"Processing class {label_idx + 1}/100: '{gloss}' ({len(entry['instances'])} videos)")
        
        for inst in entry['instances']:
            video_id = inst['video_id']
            video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
            
            if not os.path.exists(video_path):
                continue
                
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks = extractor.extract_landmarks(frame)
                frames.append(landmarks)
            cap.release()
            
            if len(frames) > 0:
                seq = np.array(frames)
                processed = process_sequence(seq)
                if processed is not None:
                    X_data.append(processed)
                    y_data.append(label_idx)
    
    extractor.close()
    
    X_arr = np.array(X_data)
    y_arr = to_categorical(y_data, num_classes=100) # One-hot encoding
    
    print("\n--- Summary ---")
    print(f"Total Samples: {len(X_arr)}")
    print(f"X shape: {X_arr.shape}")
    print(f"y shape: {y_arr.shape}")
    
    np.save(OUT_X, X_arr)
    np.save(OUT_Y, y_arr)
    print(f"Saved X data to {OUT_X}")
    print(f"Saved y data to {OUT_Y}")

if __name__ == "__main__":
    main()
