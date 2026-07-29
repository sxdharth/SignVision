import os
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

WLASL_DIR = 'Data/WLASL_Processed'
CUSTOM_DIR = 'Data/Custom_Processed'
STATIC_DIR = 'Data/Static_Processed'
OUTPUT_DIR = 'Data'
MAX_LENGTH = 30 # Max frames per sequence

def merge_data():
    raw_classes = set()
    
    # Collect all unique class names
    if os.path.exists(WLASL_DIR):
        raw_classes.update(os.listdir(WLASL_DIR))
    if os.path.exists(CUSTOM_DIR):
        raw_classes.update(os.listdir(CUSTOM_DIR))
    if os.path.exists(STATIC_DIR):
        raw_classes.update(os.listdir(STATIC_DIR))
        
    # Filter Classes (Pivot: Top Classes Only)
    MIN_SAMPLES = 10
    valid_classes = []
    
    print(f"Total raw classes: {len(raw_classes)}")
    print(f"Applying filter: Keeping classes with >= {MIN_SAMPLES} samples...")
    
    for cls in raw_classes:
        count = 0
        # Check all directories
        for data_dir in [WLASL_DIR, CUSTOM_DIR, STATIC_DIR]:
            cls_path = os.path.join(data_dir, cls)
            if os.path.exists(cls_path):
                count += len([f for f in os.listdir(cls_path) if f.endswith('.npy')])
        
        if count >= MIN_SAMPLES:
            valid_classes.append(cls)
            
    classes = sorted(valid_classes)
    class_map = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Filtered down to {len(classes)} classes (Dropped {len(raw_classes) - len(classes)} 'ghost' classes).")
    print(f"Top classes: {classes[:10]}...")
    
    X = []
    y = []
    
    # Helper to load from a directory
    def load_from_dir(base_dir):
        if not os.path.exists(base_dir):
            return
        
        for cls in os.listdir(base_dir):
            cls_dir = os.path.join(base_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            if cls not in class_map:
                continue
                
            label = class_map[cls]
            
            for file in os.listdir(cls_dir):
                if file.endswith('.npy'):
                    file_path = os.path.join(cls_dir, file)
                    sequence = np.load(file_path)
                    
                    if len(sequence) < 10:
                        continue
                        
                    X.append(sequence)
                    y.append(label)

    print("Loading WLASL data...")
    load_from_dir(WLASL_DIR)
    
    print("Loading Custom data...")
    load_from_dir(CUSTOM_DIR)

    print("Loading Static data...")
    load_from_dir(STATIC_DIR)
    
    if len(X) == 0:
        print("No data found!")
        return

    print(f"Total sequences before balancing: {len(X)}")
    
    # helper for data augmentation
    def add_noise(sequence, variance=0.01):
        """Adds Gaussian noise to the sequence."""
        noise = np.random.normal(0, variance, sequence.shape)
        return sequence + noise

    # Class Balancing (Oversampling with Noise)
    class_counts = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1
        
    max_samples = max(class_counts.values())
    print(f"Balancing classes to {max_samples} samples each (with Augmentation)...")
    
    X_balanced = []
    y_balanced = []
    
    # Group data by class
    data_by_class = {i: [] for i in range(len(classes))}
    for i, label in enumerate(y):
        data_by_class[label].append(X[i])
        
    # Oversample
    for label, samples in data_by_class.items():
        # Add original samples
        X_balanced.extend(samples)
        y_balanced.extend([label] * len(samples))
        
        # Add copies to reach max_samples
        current_count = len(samples)
        if current_count < max_samples and current_count > 0:
            needed = max_samples - current_count
            for _ in range(needed):
                # Pick a random sample to augment
                idx = np.random.randint(0, current_count)
                original_seq = samples[idx]
                
                # Create a new version with noise
                augmented_seq = add_noise(original_seq, variance=0.005)
                X_balanced.append(augmented_seq)
                y_balanced.append(label)
                
    y = np.array(y_balanced)
    
    print(f"Total sequences after balancing: {len(X_balanced)}")
    
    # Dynamically Pad and Time-Shift sequences
    print("Padding and applying Random Temporal Shifts...")
    X_final = []
    
    for seq in X_balanced:
        # Create an empty 30-frame array of zeros
        padded_seq = np.zeros((MAX_LENGTH, seq.shape[1]), dtype='float32')
        
        seq_len = min(len(seq), MAX_LENGTH)
        # Randomly choose a start index (temporal shift) to simulate live webcam delay
        max_start_idx = MAX_LENGTH - seq_len
        start_idx = np.random.randint(0, max_start_idx + 1) if max_start_idx > 0 else 0
        
        # Place the gesture inside the 30-frame window at the random offset
        padded_seq[start_idx : start_idx + seq_len] = seq[:seq_len]
        X_final.append(padded_seq)
        
    X = np.array(X_final)
    y = to_categorical(y_balanced, num_classes=len(classes))
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_combined.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_combined.npy'), y)
    
    with open(os.path.join(OUTPUT_DIR, 'combined_classes.json'), 'w') as f:
        json.dump(class_map, f)
        
    print("Data merging complete.")

if __name__ == "__main__":
    merge_data()
