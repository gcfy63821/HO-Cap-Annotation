import trimesh
import numpy as np
import yaml

def load_camera_parameters(camera_params_file):
    """
    从相机参数文件加载相机坐标变换矩阵，忽略tag_0和tag_1。
    假设文件结构包含多个相机外参
    """
    # 读取 YAML 文件
    with open(camera_params_file, 'r') as f:
        camera_params = yaml.safe_load(f)
    
    # 过滤掉 tag_0 和 tag_1
    filtered_params = {key: value for key, value in camera_params['extrinsics'].items() if not key.startswith('tag')}
    
    camera_to_world = {}
    camera_positions = {}  # 存储相机的位置
    
    for cam, extrinsics in filtered_params.items():
        # 将外参从列表转换为 4x4 矩阵
        mat = np.array(extrinsics).reshape(3, 4)
        # 添加一个用于 4x4 变换的齐次坐标行
        transform = np.vstack([mat, [0, 0, 0, 1]])
        camera_to_world[cam] = transform
        
        # 提取相机的平移部分作为相机位置
        camera_position = transform[:3, 3]  # 平移向量是变换矩阵的最后一列
        camera_positions[cam] = camera_position
    
    return camera_to_world, camera_positions

def visualize_camera_positions(camera_params_file):
    """
    加载相机参数文件，检查相机位置并在3D空间中可视化
    """
    # 加载相机的坐标变换矩阵
    camera_to_world, camera_positions = load_camera_parameters(camera_params_file)
    
    # 创建场景
    scene = trimesh.Scene()
    
    # 在场景中为每个相机添加坐标系
    for cam_idx, (cam_serial, cam_transform) in enumerate(camera_to_world.items()):
        print(f"Camera {cam_serial} to World Transformation Matrix:\n{cam_transform}")
        
        # 创建坐标轴
        coordinate_frame_vis = trimesh.creation.axis()
        coordinate_frame_vis.apply_transform(cam_transform)
        
        # 将坐标轴添加到场景中
        scene.add_geometry(coordinate_frame_vis)
        
        # 可视化相机位置（用点表示）
        camera_position = camera_positions[cam_serial]
        print(f"Camera {cam_serial} Position: {camera_position}")
        
        # 在相机位置添加一个点
        camera_point = trimesh.creation.uv_sphere(radius=0.05)  # 使用小球来表示相机位置
        camera_point.apply_translation(camera_position)
        scene.add_geometry(camera_point)
    
    # 显示场景
    scene.show()

# 示例使用
if __name__ == "__main__":
    camera_params_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    
    # 可视化相机位置
    visualize_camera_positions(camera_params_file)
