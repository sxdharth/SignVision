import numpy as np
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

X = np.load(r'd:\SignVision_S8_V2\Data\X_video_call.npy')
y = np.load(r'd:\SignVision_S8_V2\Data\y_video_call.npy')
c = json.load(open(r'd:\SignVision_S8_V2\Data\video_call_classes.json'))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

inv = {v: k for k, v in c.items()}
u, cnt = np.unique(y, return_counts=True)
print("Class distribution:")
for i, n in zip(u, cnt):
    print(f"  {inv.get(int(i), i)}: {n} samples")
