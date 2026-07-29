import os
import numpy as np
import json
import random

# Configuration
MAX_SEQ_LENGTH = 30 # standard sequence length

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
CUSTOM_PROCESSED_DIR = os.path.join(DATA_DIR, "Custom_Processed")

OUT_X = os.path.join(DATA_DIR, "X_video_call.npy")
OUT_Y = os.path.join(DATA_DIR, "y_video_call.npy")
OUT_CLASSES = os.path.join(DATA_DIR, "video_call_classes.json")

def process_sequence(seq):
    if len(seq) == 0:
        return [None]
        
    feature_size = seq.shape[1] if len(seq.shape) > 1 else seq[0].shape[0]
    sequences = []
    
    if len(seq) > MAX_SEQ_LENGTH:
        indices = np.linspace(0, len(seq) - 1, MAX_SEQ_LENGTH, dtype=int)
        base_seq = seq[indices]
        sequences.append(base_seq)
    elif len(seq) < MAX_SEQ_LENGTH:
        pad_amount = MAX_SEQ_LENGTH - len(seq)
        
        # Generate temporal shifts to teach the AI time-independence
        shifts_to_make = [0, pad_amount // 2, pad_amount] if pad_amount > 2 else [0, pad_amount]
        
        # Add a couple of random shifts to maximize variance
        for _ in range(2):
            shifts_to_make.append(np.random.randint(0, pad_amount + 1))
                
        # Create augmented copies
        for offset in set(shifts_to_make):
            padded_seq = np.zeros((MAX_SEQ_LENGTH, feature_size), dtype='float32')
            padded_seq[offset : offset + len(seq)] = seq
            
            # Inject microscopic noise to prevent identical mathematical overfitting
            noise = np.random.normal(0, 0.002, padded_seq.shape)
            padded_seq = padded_seq + noise
            
            sequences.append(padded_seq)
            
    else:
        sequences.append(seq)
        
    return sequences

def load_samples_from_dir(class_name, source_dir):
    class_dir = os.path.join(source_dir, class_name)
    samples = []
    if os.path.exists(class_dir):
        for file in os.listdir(class_dir):
            if file.endswith('.npy'):
                path = os.path.join(class_dir, file)
                try:
                    data = np.load(path)
                    processed_list = process_sequence(data)
                    for processed in processed_list:
                        if processed is not None:
                            samples.append(processed)
                except Exception as e:
                    print(f"    [Warning] Skipping corrupted file: {file} - {e}")
    return samples

def main():
    X_data = []
    y_data = []
    
    # Dynamically find classes with at least 10 samples
    valid_classes = []
    for folder in os.listdir(CUSTOM_PROCESSED_DIR):
        folder_path = os.path.join(CUSTOM_PROCESSED_DIR, folder)
        if os.path.isdir(folder_path):
            npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            if len(npy_files) > 10:
                valid_classes.append(folder)
                
    valid_classes = sorted(valid_classes)
    class_map = {cls: idx for idx, cls in enumerate(valid_classes)}
    
    print(f"Dynamic Video Call Class Map: {class_map}")

    # Load Classes
    for cls in valid_classes:
        print(f"Loading '{cls}'...")
        samples = load_samples_from_dir(cls, CUSTOM_PROCESSED_DIR)
        print(f"  -> Found {len(samples)} valid sequences.")
        
        label_id = class_map[cls]
        for s in samples:
            X_data.append(s)
            y_data.append(label_id)

    # Save
    if len(X_data) == 0:
        print("Error: No data found to merge!")
        return
        
    X_arr = np.array(X_data)
    y_arr = np.array(y_data)
    
    print("\n--- Summary ---")
    print(f"Total Samples: {len(X_arr)}")
    print(f"X shape: {X_arr.shape}")
    print(f"y shape: {y_arr.shape}")
    
    np.save(OUT_X, X_arr)
    np.save(OUT_Y, y_arr)
    print(f"Saved X data to {OUT_X}")
    print(f"Saved y data to {OUT_Y}")
    
    with open(OUT_CLASSES, 'w') as f:
        json.dump(class_map, f, indent=4)
    print(f"Saved classes to {OUT_CLASSES}")

if __name__ == "__main__":
    main()
