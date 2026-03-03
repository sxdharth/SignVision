import json
import os

def export_dictionary():
    json_path = 'Data/combined_classes.json'
    output_path = 'docs/Vocabulary_with_IDs.txt'
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    try:
        with open(json_path, 'r') as f:
            class_map = json.load(f)
            
        # Custom Sorting: Numbers -> Single Letters -> Words
        def sort_key(word):
            if word.isdigit():
                return (0, int(word))
            elif len(word) == 1:
                return (1, word)
            else:
                return (2, word)
                
        words = sorted(list(class_map.keys()), key=sort_key)
        
        with open(output_path, 'w') as f:
            f.write("SignVision AI - Supported Vocabulary\n")
            f.write("====================================\n")
            f.write(f"Total Signs: {len(words)}\n")
            f.write("Format: [Class ID] Word\n\n")
            
            for word in words:
                class_id = class_map[word]
                f.write(f"[{class_id}] {word.title()}\n")
                
        print(f"Successfully exported {len(words)} words to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    export_dictionary()
