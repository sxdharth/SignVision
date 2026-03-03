import json
import os
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

WLASL_JSON_PATH = 'asl_dataset_video/WLASL_v0.3.json'
VIDEOS_DIR = 'asl_dataset_video/videos'
OUTPUT_DIR = 'Data/WLASL_Processed'

# Set to True to process ALL classes in the dataset (Warning: Time consuming!)
PROCESS_ALL = True

# Define the classes we want to extract from WLASL
# You can add more classes to this list
SELECTED_CLASSES = ['hello', 'thanks', 'no', 'yes', 'please', 'help', 'good', 'bad', 'like', 'want']

def process_wlasl():
    if not os.path.exists(WLASL_JSON_PATH):
        print(f"Error: {WLASL_JSON_PATH} not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading WLASL JSON from {WLASL_JSON_PATH}...")
    with open(WLASL_JSON_PATH, 'r') as f:
        content = json.load(f)

    extractor = FeatureExtractor()
    
    # Create a map for faster lookup if needed, or just iterate
    # WLASL content is a list of dicts: [{'gloss': 'book', 'instances': [...]}, ...]
    
    processed_count = 0
    
    for entry in content:
        gloss = entry['gloss']
        if PROCESS_ALL or gloss in SELECTED_CLASSES:
            print(f"Processing class: {gloss}")
            class_dir = os.path.join(OUTPUT_DIR, gloss)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                
            for inst in entry['instances']:
                video_id = inst['video_id']
                video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
                
                if not os.path.exists(video_path):
                    # Try .mkv or other extensions if needed, but WLASL is mostly mp4
                    # Some might be missing
                    continue
                
                save_path = os.path.join(class_dir, f"{video_id}.npy")
                if os.path.exists(save_path):
                    print(f"Skipping {video_id}, already processed.")
                    continue
                
                print(f"  Processing video: {video_id}")
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
                    np.save(save_path, np.array(frames))
                    processed_count += 1
                
    extractor.close()
    print(f"Done. Processed {processed_count} videos.")

if __name__ == "__main__":
    process_wlasl()
