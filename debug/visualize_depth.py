import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_depth_png(depth_path, output_jpg_path=None, percentile_range=(5, 95)):
    # 读取 16-bit 深度图（通常为毫米）
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Cannot load image: {depth_path}")
    if depth_raw.dtype != np.uint16:
        raise ValueError("Expected depth image to be 16-bit (uint16) PNG.")

    print(f"[INFO] Depth image shape: {depth_raw.shape}, dtype: {depth_raw.dtype}")
    print(f"[INFO] Min: {depth_raw.min()}, Max: {depth_raw.max()}, Mean: {depth_raw.mean()}")

    # 创建可视化图像（伪彩色）
    depth = depth_raw.astype(np.float32)
    valid_mask = depth > 0

    if not np.any(valid_mask):
        raise ValueError("No valid depth values found.")

    # 自动剪裁上下限（排除极端值）
    vmin, vmax = np.percentile(depth[valid_mask], percentile_range)
    depth_vis = np.clip((depth - vmin) / (vmax - vmin), 0, 1)  # 归一化到 [0,1]

    # 应用伪彩色映射（jet）
    depth_colored = plt.cm.jet(depth_vis)[:, :, :3]  # RGB
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # 显示
    plt.imshow(depth_colored)
    plt.title("Visualized Depth")
    plt.axis('off')
    plt.show()

    # 保存为 .jpg（可选）
    if output_jpg_path:
        os.makedirs(os.path.dirname(output_jpg_path), exist_ok=True)
        cv2.imwrite(output_jpg_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved visualized depth to {output_jpg_path}")


# 示例调用
if __name__ == "__main__":
    depth_png_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_203130/07/depth_000100.png"  # 替换为你的路径
    output_jpg_path = "debug_outputs/depth_vis.jpg"        # 保存可视化结果的路径（可选）
    visualize_depth_png(depth_png_path, output_jpg_path)
