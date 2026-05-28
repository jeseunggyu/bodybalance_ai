"""데이터 구조 진단"""
import numpy as np
import pandas as pd
from pathlib import Path

sample = Path("data/raw/public/P001/W1_BF")

# npz 구조 확인
npz = np.load(sample / "pipeline_1.npz")
print("=" * 50)
print("pipeline_1.npz 구조")
print("=" * 50)
print(f"키 목록: {list(npz.files)}")
for key in npz.files:
    arr = npz[key]
    print(f"  '{key}': shape={arr.shape}, dtype={arr.dtype}")
    print(f"    min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.4f}")

# metadata.csv 구조 확인
print("\n" + "=" * 50)
print("metadata.csv 구조")
print("=" * 50)
meta = pd.read_csv(sample / "metadata.csv")
print(f"행 수: {len(meta)}")
print(f"컬럼 목록:")
for col in meta.columns:
    print(f"  - {col}: {meta[col].dtype}")
print(f"\n처음 3행:")
print(meta.head(3).to_string())