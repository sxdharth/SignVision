import numpy as np
import json
import os

# Paths to the Custom Processed Data
X_PATH = os.path.join("Data", "Custom_Processed", "X_custom.npy")
Y_PATH = os.path.join("Data", "Custom_Processed", "y_custom.npy")
CLASSES_PATH = os.path.join("Data", "Custom_Processed", "custom_classes.json")

def remove_smart_home_data():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH) or not os.path.exists(CLASSES_PATH):
        print("Custom dataset files not found. Nothing to delete.")
        return

    # Load data
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    with open(CLASSES_PATH, 'r') as f:
        classes_map = json.load(f)

    # Invert mapping to read labels
    idx_to_class = {v: k for k, v in classes_map.items()}

    # Targeted classes to delete
    targets = ["bulb_on", "bulb_off"]
    target_indices = [classes_map.get(t) for t in targets if t in classes_map]
    
    # Filter out None values in case a target class doesn't exist
    target_indices = [idx for idx in target_indices if idx is not None]

    if not target_indices:
        print("Notice: 'bulb_on' and 'bulb_off' are already not present in the dataset.")
        return

    print(f"Original dataset shape: {X.shape}")
    print(f"Deleting sequences for: {[idx_to_class[i] for i in target_indices]}")

    # Keep only indices where the corresponding ground truth label is NOT in target_indices
    keep_mask = []
    # y is one-hot encoded (e.g., [0, 0, 1, 0...]) so we use argmax to get the index integer
    for label_one_hot in y:
        class_idx = np.argmax(label_one_hot)
        keep_mask.append(class_idx not in target_indices)

    # Convert to numpy array for indexing
    keep_mask = np.array(keep_mask)

    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]

    # Re-save the filtered dataset, overwriting the old one
    np.save(X_PATH, X_filtered)
    np.save(Y_PATH, y_filtered)
    
    print(f"Success! Filtered dataset shape: {X_filtered.shape}")
    num_deleted = len(X) - len(X_filtered)
    print(f"Deleted {num_deleted} total '{targets[0]}' and '{targets[1]}' samples.")

if __name__ == "__main__":
    remove_smart_home_data()
