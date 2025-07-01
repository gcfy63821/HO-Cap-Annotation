import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_depth_image(path):
    """
    读取深度 PNG 图像，支持 uint8 和 uint16。
    """
    img = Image.open(path)
    depth = np.array(img)

    info = {
        'path': path,
        'shape': depth.shape,
        'dtype': depth.dtype,
        'min': np.min(depth),
        'max': np.max(depth),
        'mean': np.mean(depth),
        'nan_ratio': np.isnan(depth).sum() / depth.size
    }
    return depth, info


def analyze_depth_folder(folder, limit=5, show_images=True):
    """
    分析文件夹中的若干张 PNG 深度图。
    """
    all_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".png")
    ])

    print(f"[INFO] Found {len(all_files)} PNG files in {folder}")
    for idx, file_path in enumerate(all_files[:limit]):
        depth, info = read_depth_image(file_path)
        print(f"\n[Depth Image {idx+1}]")
        for k, v in info.items():
            print(f"  {k}: {v}")

        if show_images:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(depth, cmap='gray')
            plt.title("Raw Depth (Grayscale)")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.hist(depth.ravel(), bins=100)
            plt.title("Depth Histogram")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # 👉 修改为你真实数据集深度图所在的路径
    depth_image_folder = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/datasets/subject_1/20231025_165502/037522251142"

    # 开始分析
    analyze_depth_folder(depth_image_folder, limit=3, show_images=True)
