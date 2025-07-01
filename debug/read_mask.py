from pathlib import Path
import cv2
import numpy as np

mask_root = Path("crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250624_140312/processed/segmentation/sam2")
num_invalid = 0

for mask_path in sorted(mask_root.rglob("mask_*.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if np.sum(mask) == 0:
        print(f"[WARNING] Empty mask: {mask_path}")
        num_invalid += 1

print(f"\nTotal empty masks: {num_invalid}")
