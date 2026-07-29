import os
import numpy as np
import json
import random

# Configuration
SMART_HOME_CLASSES = ["bulb_off", "bulb_on"] 
MAX_SEQ_LENGTH = 30 # standard sequence length

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "Data")
CUSTOM_PROCESSED_DIR = os.path.join(DATA_DIR, "Custom_Processed")
STATIC_PROCESSED_DIR = os.path.join(DATA_DIR, "Static_Processed")

OUT_X = os.path.join(DATA_DIR, "X_smart_home.npy")
OUT_Y = os.path.join(DATA_DIR, "y_smart_home.npy")
OUT_CLASSES = os.path.join(DATA_DIR, "smart_home_classes.json")

def process_sequence(seq):
    """Pads or truncates a sequence to MAX_SEQ_LENGTH frames.
    Also returns slightly shifted copies for data augmentation."""
    if len(seq) == 0:
        return [None]
        
    # Get feature size from first frame
    feature_size = seq.shape[1] if len(seq.shape) > 1 else seq[0].shape[0]
    
    sequences = []
    
    # 1. Base sequence
    if len(seq) > MAX_SEQ_LENGTH:
        # Evenly sample MAX_SEQ_LENGTH frames across the entire sequence
        indices = np.linspace(0, len(seq) - 1, MAX_SEQ_LENGTH, dtype=int)
        base_seq = seq[indices]
    elif len(seq) < MAX_SEQ_LENGTH:
        pad_amount = MAX_SEQ_LENGTH - len(seq)
        padding = np.zeros((pad_amount, feature_size))
        base_seq = np.vstack((seq, padding))
    else:
        base_seq = seq
    sequences.append(base_seq)
        
    return sequences

def process_sequence_single(seq, feature_size):
    if len(seq) > MAX_SEQ_LENGTH:
        indices = np.linspace(0, len(seq) - 1, MAX_SEQ_LENGTH, dtype=int)
        return seq[indices]
    elif len(seq) < MAX_SEQ_LENGTH:
        pad_amount = MAX_SEQ_LENGTH - len(seq)
        padding = np.zeros((pad_amount, feature_size))
        return np.vstack((seq, padding))
    return seq

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

def generate_background_samples(num_samples):
    print(f"Generating {num_samples} background ('none') samples...")
    all_npy_files = []
    
    for d in [CUSTOM_PROCESSED_DIR, STATIC_PROCESSED_DIR]:
        if os.path.exists(d):
            for root, dirs, files in os.walk(d):
                folder_name = os.path.basename(root)
                if folder_name not in SMART_HOME_CLASSES:
                    for f in files:
                        if f.endswith('.npy'):
                            all_npy_files.append(os.path.join(root, f))
                            
    if not all_npy_files:
        return []
        
    selected_files = random.sample(all_npy_files, min(num_samples, len(all_npy_files)))
    
    samples = []
    for f in selected_files:
        try:
            data = np.load(f)
            processed_list = process_sequence(data)
            # For background, we just need one of the sequences to not overinflate
            if processed_list and processed_list[0] is not None:
                samples.append(processed_list[0])
        except Exception as e:
            print(f"    [Warning] Skipping corrupted background file: {f} - {e}")
            
    print(f"    -> Harvested {len(samples)} real idle sequences from background datasets.")
    return samples

def main():
    X_data = []
    y_data = []
    
    class_map = {"none": 0}
    for idx, cls in enumerate(sorted(SMART_HOME_CLASSES)):
        class_map[cls] = idx + 1
        
    print(f"Smart Home Class Map: {class_map}")

    # 1. Load Smart Home Classes
    max_class_samples = 0
    for cls in SMART_HOME_CLASSES:
        print(f"Loading '{cls}'...")
        samples = load_samples_from_dir(cls, CUSTOM_PROCESSED_DIR)
        print(f"  -> Found {len(samples)} valid sequences.")
        valid_len = len(samples)
        if valid_len > max_class_samples:
            max_class_samples = valid_len
        
        label_id = class_map[cls]
        for s in samples:
            X_data.append(s)
            y_data.append(label_id)

    # 2. Add background data
    # To prevent false positives, we must train heavily on ACTUAL idle videos from the Static dataset,
    # NOT mathematical zeros, because mediapipe still outputs actual shoulders/face data when idle.
    num_bg = max(150, int(max_class_samples * 6))
    bg_samples = generate_background_samples(num_bg)
    print(f"  -> Found {len(bg_samples)} valid background sequences.")
    for s in bg_samples:
        X_data.append(s)
        y_data.append(class_map["none"])
        
    # 3. Save
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
