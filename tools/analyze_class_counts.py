import os
import glob
import matplotlib.pyplot as plt

def analyze_counts():
    data_dir = 'Data/WLASL_Processed'
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    print("Counting samples per class...")
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    counts = []
    class_stats = {}
    
    for cls in class_dirs:
        # Count original npy files (assuming standard naming or just checking count)
        # Note: augmentation might verify later, just counting files in processed dir
        files = glob.glob(os.path.join(data_dir, cls, '*.npy'))
        count = len(files)
        counts.append(count)
        class_stats[cls] = count
        
    counts.sort(reverse=True)
    
    output_path = 'docs/Dataset_Analysis.txt'
    with open(output_path, 'w') as f:
        f.write("--- DATA DISTRIBUTION REPORT ---\n")
        f.write(f"Total Classes: {len(class_dirs)}\n")
        f.write(f"Max samples in a class: {max(counts)}\n")
        f.write(f"Min samples in a class: {min(counts)}\n")
        f.write(f"Average samples: {sum(counts)/len(counts):.1f}\n\n")
        
        # Stratification buckets
        over_100 = sum(1 for c in counts if c >= 100)
        over_50 = sum(1 for c in counts if c >= 50)
        over_20 = sum(1 for c in counts if c >= 20)
        under_10 = sum(1 for c in counts if c < 10)
        
        f.write("--- VIABILITY CHECK ---\n")
        f.write(f"Classes with > 100 samples (Excellent): {over_100}\n")
        f.write(f"Classes with > 50 samples  (Good):      {over_50}\n")
        f.write(f"Classes with > 20 samples  (Okay):      {over_20}\n")
        f.write(f"Classes with < 10 samples  (Garbage):   {under_10}\n\n")
        
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        f.write("Top 50 Classes by Data Volume:\n")
        for k, v in sorted_classes[:50]:
            f.write(f"  {k}: {v}\n")
            
        f.write("\nBottom 20 Classes:\n")
        for k, v in sorted_classes[-20:]:
            f.write(f"  {k}: {v}\n")
            
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    analyze_counts()
