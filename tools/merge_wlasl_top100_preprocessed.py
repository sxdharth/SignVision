import os
import numpy as np
import json
from tensorflow.keras.utils import to_categorical

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
WLASL_DIR = os.path.join(DATA_DIR, "WLASL_Processed")

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
    print("Scanning WLASL_Processed directory...")
    class_counts = []
    
    for cls in os.listdir(WLASL_DIR):
        cls_path = os.path.join(WLASL_DIR, cls)
        if os.path.isdir(cls_path):
            npy_files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            if len(npy_files) > 0:
                class_counts.append((cls, len(npy_files)))
                
    # Sort descending by count
    class_counts.sort(key=lambda x: x[1], reverse=True)
    top_100 = class_counts[:100]
    
    print(f"Selected Top {len(top_100)} classes.")
    
    class_map = {item[0]: idx for idx, item in enumerate(top_100)}
    with open(OUT_CLASSES, 'w') as f:
        json.dump(class_map, f, indent=4)
        
    X_data = []
    y_data = []
    
    for cls, count in top_100:
        cls_path = os.path.join(WLASL_DIR, cls)
        label_idx = class_map[cls]
        
        npy_files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
        for npy_file in npy_files:
            path = os.path.join(cls_path, npy_file)
            try:
                seq = np.load(path)
                processed = process_sequence(seq)
                if processed is not None:
                    X_data.append(processed)
                    y_data.append(label_idx)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                
    X_arr = np.array(X_data)
    y_arr = to_categorical(y_data, num_classes=100)
    
    print("\n--- Summary ---")
    print(f"Total Samples Extract: {len(X_arr)}")
    print(f"X shape: {X_arr.shape}")
    print(f"y shape: {y_arr.shape}")
    print(f"Dataset contains top {len(top_100)} classes ranging from {top_100[-1][1]} to {top_100[0][1]} videos per class.")
    
    np.save(OUT_X, X_arr)
    np.save(OUT_Y, y_arr)
    print(f"Saved X data to {OUT_X}")
    print(f"Saved y data to {OUT_Y}")

if __name__ == "__main__":
    main()
