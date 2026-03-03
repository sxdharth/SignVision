import numpy as np
import os
import glob

def check_stats():
    # Find a few npy files
    files = glob.glob('Data/WLASL_Processed/*/*.npy')
    if not files:
        print("No NPY files found!")
        return

    print(f"Found {len(files)} files. Checking first 5...")
    
    for fpath in files[:5]:
        data = np.load(fpath)
        print(f"\nFile: {fpath}")
        print(f"Shape: {data.shape}")
        print(f"Min: {np.min(data)}, Max: {np.max(data)}")
        print(f"Mean: {np.mean(data)}")
        if np.all(data == 0):
            print("WARNING: DATA IS ALL ZEROS")
        else:
            print("Data looks valid (non-zero).")

if __name__ == "__main__":
    check_stats()
