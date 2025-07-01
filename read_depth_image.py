import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import imageio


def analyze_depth_pngs(depth_folder, visualize_one=True):
    depth_folder = Path(depth_folder)
    png_files = sorted(depth_folder.glob("*.png"))

    if not png_files:
        print(f"[ERROR] No PNG files found in {depth_folder}")
        return

    print(f"[INFO] Found {len(png_files)} depth PNG files.")

    stats = []

    for i, png_file in enumerate(png_files):
        depth_img = Image.open(png_file)
        depth_array = np.array(depth_img)

        # 数据类型和维度
        print(f"\n[File {i+1}] {png_file.name}")
        print(f" - Shape: {depth_array.shape}")
        print(f" - Dtype: {depth_array.dtype}")

        # 深度值范围
        print(f" - Min: {depth_array.min()}, Max: {depth_array.max()}, Mean: {depth_array.mean():.2f}")

        stats.append((depth_array.min(), depth_array.max(), depth_array.mean()))

        if visualize_one:
            plt.imshow(depth_array, cmap='plasma')
            plt.title(f"{png_file.name} (depth)")
            plt.colorbar(label="Depth value")
            plt.show()
            visualize_one = False  # 只显示一张

    # 汇总统计
    all_mins, all_maxs, all_means = zip(*stats)
    print("\n====== Summary ======")
    print(f" - Overall Min: {min(all_mins)}")
    print(f" - Overall Max: {max(all_maxs)}")
    print(f" - Mean of Means: {np.mean(all_means):.2f}")


# 示例调用
if __name__ == "__main__":
    # 修改为你的深度图文件夹路径
    depth_dir = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/datasets/subject_1/20231025_165502/043422252387"
    # analyze_depth_pngs(depth_dir)

    depth = imageio.v2.imread(depth_dir + "/depth_000000.png")  # 读取为数组
    print("Shape:", depth.shape)
    print("Dtype:", depth.dtype)

