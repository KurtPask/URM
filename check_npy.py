# check_npy_size.py
from pathlib import Path
import numpy as np
import sys

# pass the path like: python check_npy_size.py subdir/myfile.npy
npy_path = Path(sys.argv[1]).expanduser().resolve()

if not npy_path.exists():
    raise FileNotFoundError(f"File not found: {npy_path}")

# size on disk
disk_size_bytes = npy_path.stat().st_size

# load header + array metadata without fully reading into RAM
arr = np.load(npy_path, mmap_mode="r")

print(f"Path: {npy_path}")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Array size in memory: {arr.nbytes:,} bytes ({arr.nbytes / 1024**3:.3f} GB)")
print(f"File size on disk:    {disk_size_bytes:,} bytes ({disk_size_bytes / 1024**3:.3f} GB)")