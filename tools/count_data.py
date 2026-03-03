import json

WLASL_JSON_PATH = 'asl_dataset_video/WLASL_v0.3.json'

try:
    with open(WLASL_JSON_PATH, 'r') as f:
        content = json.load(f)
        
    num_classes = len(content)
    total_instances = sum(len(entry['instances']) for entry in content)
    
    print(f"Total Classes: {num_classes}")
    print(f"Total Video Instances: {total_instances}")
    
except Exception as e:
    print(f"Error reading JSON: {e}")
