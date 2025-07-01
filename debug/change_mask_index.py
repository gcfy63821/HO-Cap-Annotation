import numpy as np
import cv2

def convert_mask_labels(input_file: str, output_file: str) -> None:
    """
    读取 PNG mask 文件，替换标签，将物体标签（255）替换为 1，背景标签保持为 0，
    然后保存新的 mask 文件。
    
    :param input_file: 输入的 PNG mask 文件路径
    :param output_file: 输出的新的 PNG mask 文件路径
    """
    # 读取 PNG mask 图像
    mask_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)  # 使用灰度模式读取
    
    # 检查图像是否正确读取
    if mask_image is None:
        print("Error: Unable to read the mask image.")
        return
    
    # 替换标签：将标签 255 替换为 1，背景 0 保持不变
    mask_image[mask_image == 255] = 1
    
    # 保存新的 mask 文件
    cv2.imwrite(output_file, mask_image)
    print(f"Mask file saved as {output_file}")

# 使用示例
input_file = '/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250629_164925/processed/segmentation/init/00/mask_000000.png'  # 输入文件路径
output_file = '/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250629_164925/processed/segmentation/init/00/mask_000000.png'  # 输出文件路径
convert_mask_labels(input_file, output_file)
