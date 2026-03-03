import json
import os

def export_video_map():
    json_path = 'asl_dataset_video/WLASL_v0.3.json'
    output_path = 'docs/Word_to_Video_Map.txt'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    try:
        print(f"Loading WLASL data from {json_path}...")
        with open(json_path, 'r') as f:
            content = json.load(f)
            
        # Create a dictionary: Word -> [List of Video IDs]
        word_map = {}
        for entry in content:
            word = entry['gloss']
            video_ids = [inst['video_id'] for inst in entry['instances']]
            word_map[word] = video_ids

        # Custom Sorting: Numbers -> Single Letters -> Words
        def sort_key(word):
            if word.isdigit():
                return (0, int(word))
            elif len(word) == 1:
                return (1, word)
            else:
                return (2, word)
                
        sorted_words = sorted(list(word_map.keys()), key=sort_key)
        
        print(f"Writing mapping to {output_path}...")
        with open(output_path, 'w') as f:
            f.write("SignVision AI - Word to Video ID Mapping\n")
            f.write("========================================\n")
            f.write(f"Total Unique Words: {len(sorted_words)}\n")
            f.write("Format: Word : [Video ID 1, Video ID 2, ...]\n")
            f.write("Note: These IDs correspond to filenames in 'asl_dataset_video/videos'\n\n")
            
            for word in sorted_words:
                ids = word_map[word]
                id_str = ", ".join(ids)
                f.write(f"{word.title()} : [{id_str}]\n")
                
        print(f"Successfully exported mapping for {len(sorted_words)} words.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    export_video_map()
