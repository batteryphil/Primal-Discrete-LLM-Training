import numpy as np
import os
import glob

def inspect_mel(mels_dir):
    files = glob.glob(os.path.join(mels_dir, "*.npy"))
    if not files:
        print("No mel files found.")
        return
    
    data = np.load(files[0])
    print(f"File: {files[0]}")
    print(f"Shape: {data.shape}")
    print(f"Min: {data.min()}")
    print(f"Max: {data.max()}")
    print(f"Mean: {data.mean()}")
    print(f"First 5x5: \n{data[:5, :5]}")

if __name__ == "__main__":
    inspect_mel("data/vctk_p225/mels")
