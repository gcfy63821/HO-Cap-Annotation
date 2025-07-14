import trimesh
import numpy as np
import yaml
import os
import glob

def load_camera_parameters(camera_params_file):
    """
    从相机参数文件加载相机坐标变换矩阵 忽略tag_0和tag_1。
    假设文件结构包含多个相机外参
    """
    # 读取 YAML 文件
    with open(camera_params_file, 'r') as f:
        camera_params = yaml.safe_load(f)
    
    # 过滤掉 tag_0 和 tag_1
    filtered_params = {key: value for key, value in camera_params['extrinsics'].items() if not key.startswith('tag')}
    
    camera_to_world = {}
    for cam, extrinsics in filtered_params.items():
        # 将外参从列表转换为 4x4 矩阵
        mat = np.array(extrinsics).reshape(3, 4)
        # 添加一个用于 4x4 变换的齐次坐标行
        transform = np.vstack([mat, [0, 0, 0, 1]])
        camera_to_world[cam] = transform
        
    return camera_to_world

def visualize_mesh_and_camera(mesh_file, camera_params_file, transform_obj_in_world=None, show_texture=True):
    """
    加载网格和相机坐标变换，并显示相机位置和网格。
    show_texture: 是否显示mesh的贴图（如果有）
    """
    # 加载相机的坐标变换矩阵
    camera_to_world = load_camera_parameters(camera_params_file)
    
    # 创建场景
    scene = trimesh.Scene()
    
    # 加载网格文件
    mesh = trimesh.load(mesh_file, process=False)  # process=False 保留贴图信息
    print(f"Loaded Mesh: {mesh_file}")
    print(f"Mesh vertices: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    # 检查贴图信息
    has_texture = False
    if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
        # 检查是否有贴图文件引用
        if hasattr(mesh.visual.material, "image") and mesh.visual.material.image is not None:
            has_texture = True
    # 兼容trimesh 4.x的贴图检测
    if hasattr(mesh.visual, "to_color") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        if hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "image", None) is not None:
            has_texture = True

    if show_texture and has_texture:
        print(f"[INFO] Mesh has texture: {getattr(mesh.visual.material, 'image', None)}")
    else:
        print("[INFO] Mesh has no texture or show_texture=False.")
        # 尝试手动加载mtl和jpg
        mtl_path = mesh_file.replace('.obj', '.mtl')
        if os.path.exists(mtl_path):
            print(f"[INFO] Found mtl file: {mtl_path}")
        else:
            print(f"[WARN] No mtl file found for {mesh_file}")
        # 检查同目录下是否有jpg/png
        dir_path = os.path.dirname(mesh_file)
        img_files = glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.png"))
        if img_files:
            print(f"[INFO] Found possible texture images: {img_files}")
        else:
            print(f"[WARN] No texture images found near {mesh_file}")

    # 如果有贴图且show_texture为True，显示贴图
    if show_texture and hasattr(mesh.visual, "material") and mesh.visual.material is not None:
        print(f"[INFO] Mesh has texture: {getattr(mesh.visual.material, 'image', None)}")
    else:
        print("[INFO] Mesh has no texture or show_texture=False.")

    # 如果提供了物体的世界坐标系变换，应用物体的坐标变换
    if transform_obj_in_world is not None:
        mesh.apply_transform(transform_obj_in_world)
        print(f"Applied Object Transformation:\n{transform_obj_in_world}")
    
    # 在场景中添加网格
    scene.add_geometry(mesh)

    # 在场景中为每个相机添加坐标系
    for cam_idx, (cam_serial, cam_transform) in enumerate(camera_to_world.items()):
        # print(f"Camera {cam_serial} to World Transformation Matrix:\n{cam_transform}")
        
        # 创建坐标轴
        coordinate_frame_vis = trimesh.creation.axis()
        coordinate_frame_vis.apply_transform(cam_transform)
        
        # 将坐标轴添加到场景中
        scene.add_geometry(coordinate_frame_vis)
    
    # 显示场景
    scene.show()

# 示例使用
if __name__ == "__main__":
    # mesh_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/models/blue_scooper/cleaned_mesh_10000.obj"
    # mesh_file = "/home/wys/learning-compliant/crq_ws/data/decim_mesh_files_1/decim_mesh_files/blue_scooper.obj"
    # mesh_file = "/home/wys/learning-compliant/crq_ws/data/mesh/decim_mesh_files/blue_spoon.obj"
    # mesh_file = "/home/wys/learning-compliant/crq_ws/data/decim_mesh_files_1/decim_mesh_files/plastic_scoop.obj"
    mesh_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/models/wooden_spoon/cleaned_mesh_10000.obj"
    # mesh_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/models/wooden_spoon/textured_mesh.obj"
    camera_params_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"

    # 如果有物体的世界坐标系变换矩阵，可以传入
    transform_obj_in_world = np.eye(4)  # 这是一个示例，你可以替换为实际的变换矩阵
    
    # 可视化相机和网格（支持贴图显示）
    visualize_mesh_and_camera(mesh_file, camera_params_file, transform_obj_in_world, show_texture=True)
