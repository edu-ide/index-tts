#!/usr/bin/env python3
import torch

# feat2.pt 구조 확인
feat2 = torch.load('/mnt/sda1/models/IndexTTS-2/feat2.pt', map_location='cpu')
print("feat2.pt 구조:")
print(f"  Type: {type(feat2)}")
if isinstance(feat2, torch.Tensor):
    print(f"  Shape: {feat2.shape}")
    print(f"  차원 0: {feat2.shape[0]} (이게 vocab size와 관련?)")
elif isinstance(feat2, dict):
    for key, val in feat2.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")

# feat1.pt도 확인
print("\nfeat1.pt 구조:")
feat1 = torch.load('/mnt/sda1/models/IndexTTS-2/feat1.pt', map_location='cpu')
if isinstance(feat1, torch.Tensor):
    print(f"  Shape: {feat1.shape}")
elif isinstance(feat1, dict):
    for key, val in feat1.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
