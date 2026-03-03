import cv2
import os
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extractor import FeatureExtractor

DATASET_DIR = 'asl_dataset'
OUTPUT_DIR = 'Data/Static_Processed'
SEQUENCE_LENGTH = 30

def process_static_data():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    extractor = FeatureExtractor()
    processed_count = 0
    
    # Iterate through classes (0-9, a-z)
    classes = os.listdir(DATASET_DIR)
    
    for cls in classes:
        class_input_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(class_input_dir):
            continue
            
        print(f"Processing static class: {cls}")
        class_output_dir = os.path.join(OUTPUT_DIR, cls)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
            
        for img_name in os.listdir(class_input_dir):
            img_path = os.path.join(class_input_dir, img_name)
            
            # Read image
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            # Extract landmarks
            landmarks = extractor.extract_landmarks(frame)
            
            # Create a sequence by repeating the static landmarks
            # Add Gaussian noise to simulate camera jitter
            sequence = np.tile(landmarks, (SEQUENCE_LENGTH, 1))
            noise = np.random.normal(0, 0.005, sequence.shape) # Mean 0, Std 0.005
            sequence = sequence + noise
            
            # Save as .npy
            save_name = os.path.splitext(img_name)[0] + ".npy"
            save_path = os.path.join(class_output_dir, save_name)
            
            np.save(save_path, sequence)
            processed_count += 1
            
    extractor.close()
    print(f"Done. Processed {processed_count} static images.")

if __name__ == "__main__":
    process_static_data()
