import json
import os

def export_final_map():
    # 1. Load Supported Classes
    supported_classes_path = 'Data/combined_classes.json'
    if not os.path.exists(supported_classes_path):
        print(f"Error: {supported_classes_path} not found.")
        return
        
    with open(supported_classes_path, 'r') as f:
        class_map = json.load(f)
    
    # Get just the list of words (keys)
    supported_words = set(class_map.keys())
    print(f"Loaded {len(supported_words)} supported words from the trained model.")

    # 2. Load Master WLASL List
    master_json_path = 'asl_dataset_video/WLASL_v0.3.json'
    if not os.path.exists(master_json_path):
        print(f"Error: {master_json_path} not found.")
        return
        
    print(f"Loading Master WLASL List...")
    with open(master_json_path, 'r') as f:
        master_content = json.load(f)
        
    # 3. Check File Existence
    videos_dir = 'asl_dataset_video/videos'
    if not os.path.exists(videos_dir):
        print(f"Warning: {videos_dir} not found. Cannot verify file existence.")
        available_files = set()
    else:
        # Get all .mp4 files (without extension for faster lookup)
        available_files = set([f.split('.')[0] for f in os.listdir(videos_dir) if f.endswith('.mp4')])
        print(f"Found {len(available_files)} video files in {videos_dir}.")

    # 4. Build the Map
    final_map = {}
    
    for entry in master_content:
        word = entry['gloss']
        
        # Only process if this word is in our supported list
        if word in supported_words:
            valid_ids = []
            for inst in entry['instances']:
                vid_id = inst['video_id']
                # key check: Does this video ID exist as a file?
                if vid_id in available_files:
                    valid_ids.append(vid_id)
            
            # Store even if empty, so user knows we support the word (might be static data)
            final_map[word] = valid_ids

    # 5. Add words that are supported but might not be in WLASL master list (Custom/Static)
    for word in supported_words:
        if word not in final_map:
            final_map[word] = ["(Custom/Static Data - No WLASL Video)"]

    # 6. Sort and Write
    def sort_key(word):
        if word.isdigit(): return (0, int(word))
        elif len(word) == 1: return (1, word)
        else: return (2, word)
        
    sorted_words = sorted(list(final_map.keys()), key=sort_key)
    
    output_path = 'docs/Final_Vocabulary_with_Video_IDs.txt'
    print(f"Writing final map to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("SignVision AI - Final Supported Vocabulary & Video Reference\n")
        f.write("==========================================================\n")
        f.write(f"Total Supported Words: {len(sorted_words)}\n")
        f.write("Format: Word : [VideoID1, VideoID2] (Only IDs present on disk)\n\n")
        
        for word in sorted_words:
            ids = final_map[word]
            if not ids:
                id_str = "(No matching video file found in folder)"
            else:
                id_str = ", ".join(ids)
                
            f.write(f"{word} : [{id_str}]\n")
            
    print("Done.")

if __name__ == "__main__":
    export_final_map()
