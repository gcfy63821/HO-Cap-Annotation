import logging
import trimesh
import numpy as np
from hocap_annotation.utils import *
from hocap_annotation.loaders import HOCapLoader

# 配置日志
logging.basicConfig(level=logging.INFO)

sequence_folder = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/datasets/subject_1/20231025_165502"
data_loader = HOCapLoader(sequence_folder)

# 获取数据
rs_width = data_loader.rs_width
rs_height = data_loader.rs_height
num_frames = data_loader.num_frames
object_idx = 0  # 假设你选择了第一个物体，修改这个值来选择其他物体
object_id = data_loader.object_ids[object_idx]
rs_serials = data_loader.rs_serials
cam_Ks = data_loader.rs_Ks
cam_RTs = data_loader.extr2world
valid_serials = data_loader.get_valid_seg_serials()
valid_serial_indices = [rs_serials.index(serial) for serial in valid_serials]
valid_Ks = data_loader.rs_Ks[valid_serial_indices]
valid_RTs = data_loader.extr2world[valid_serial_indices]
valid_RTs_inv = data_loader.extr2world_inv[valid_serial_indices]
object_mesh_textured = trimesh.load(data_loader.object_textured_files[object_idx])
object_mesh_cleaned = trimesh.load(data_loader.object_cleaned_files[object_idx])
empty_mat_pose = np.full((4, 4), -1.0, dtype=np.float32)

# 打印数据
logging.info(f"rs_width: {rs_width}")
logging.info(f"rs_height: {rs_height}")
logging.info(f"num_frames: {num_frames}")
logging.info(f"object_id: {object_id}")
logging.info(f"rs_serials: {rs_serials}")
logging.info(f"cam_Ks: {cam_Ks.shape}")  # 打印相机内参矩阵的形状
logging.info(f"cam_RTs: {cam_RTs.shape}")  # 打印外参矩阵的形状
logging.info(f"valid_serials: {valid_serials}")
logging.info(f"valid_serial_indices: {valid_serial_indices}")
logging.info(f"valid_Ks: {valid_Ks.shape}")  # 打印有效的相机内参矩阵的形状
logging.info(f"valid_RTs: {valid_RTs.shape}")  # 打印有效外参矩阵的形状
logging.info(f"valid_RTs_inv: {valid_RTs_inv.shape}")  # 打印有效外参矩阵的逆
logging.info(f"object_mesh_textured: {object_mesh_textured}")
logging.info(f"object_mesh_cleaned: {object_mesh_cleaned}")
logging.info(f"empty_mat_pose: {empty_mat_pose}")

# Check start and end frame_idx
start_frame = 0  # 设置你的起始帧
end_frame = num_frames  # 设置你的结束帧

# 检查帧范围
start_frame = max(start_frame, 0)
end_frame = num_frames if end_frame < start_frame else end_frame

logging.info(f"start_frame: {start_frame}, end_frame: {end_frame}")
