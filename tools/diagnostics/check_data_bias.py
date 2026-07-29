import numpy as np
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the correct path considering this script is directly in the root directory for this run
DATA_DIR = os.path.join(r"d:\SignVision_S8_V2", "Data")

y_path = os.path.join(DATA_DIR, 'y_video_call.npy')
classes_path = os.path.join(DATA_DIR, 'video_call_classes.json')

try:
    y = np.load(y_path)
    with open(classes_path, 'r') as f:
        class_map = json.load(f)
        
    inverse_map = {v: k for k, v in class_map.items()}
    
    counts = {}
    for label_idx in y:
        name = inverse_map.get(label_idx, f"Unknown_{label_idx}")
        counts[name] = counts.get(name, 0) + 1
        
    print("Video Call Model Class Distribution:")
    for name, sum_count in sorted(counts.items()):
        print(f"  {name}: {sum_count} samples")
except Exception as e:
    print(f"Error checking stats: {e}")
