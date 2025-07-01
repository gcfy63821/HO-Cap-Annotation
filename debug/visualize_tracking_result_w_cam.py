import trimesh
import numpy as np
import yaml

from scipy.spatial.transform import Rotation as R


def convert_7d_pose_to_matrix_seq(pose_7d_seq):
    """
    pose_7d_seq: (N,7) numpy数组，每帧格式 [x,y,z,qx,qy,qz,qw]
    返回 (N,4,4) numpy数组的变换矩阵序列
    """
    N = pose_7d_seq.shape[0]
    matrices = np.zeros((N,4,4), dtype=np.float32)
    for i in range(N):
        t = pose_7d_seq[i, :3]
        q = pose_7d_seq[i, 3:]
        R_mat = R.from_quat(q).as_matrix()
        matrices[i] = np.eye(4)
        matrices[i][:3,:3] = R_mat
        matrices[i][:3, 3] = t
    return matrices

def load_camera_parameters(camera_params_file):
    with open(camera_params_file, 'r') as f:
        camera_params = yaml.safe_load(f)
    filtered_params = {k: v for k, v in camera_params['extrinsics'].items() if not k.startswith('tag')}
    camera_to_world = {}
    camera_positions = {}
    for cam, extrinsics in filtered_params.items():
        mat = np.array(extrinsics).reshape(3,4)
        transform = np.vstack([mat, [0,0,0,1]])
        camera_to_world[cam] = transform
        camera_positions[cam] = transform[:3, 3]
    return camera_to_world, camera_positions

def load_pose_sequence(pose_npy_path):
    poses = np.load(pose_npy_path)
    if poses.ndim == 3 and poses.shape[0] == 1 and poses.shape[2] == 7:
        poses_7d = poses[0]
        return convert_7d_pose_to_matrix_seq(poses_7d)
    else:
        raise ValueError(f"Pose npy shape expected (1,N,7), but got {poses.shape}")


def visualize_with_cameras_and_mesh(mesh_file, pose_npy_file, camera_params_file):
    # 加载相机参数
    camera_to_world, camera_positions = load_camera_parameters(camera_params_file)
    print(f"Loaded {len(camera_to_world)} cameras")

    # 加载网格
    mesh = trimesh.load(mesh_file)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

    # 加载物体的每帧位姿
    pose_sequence = load_pose_sequence(pose_npy_file)
    n_frames = pose_sequence.shape[0]
    print(f"Loaded {n_frames} poses")

    # 创建场景，先添加相机
    scene = trimesh.Scene()
    # for cam, trans in camera_to_world.items():
    #     axis = trimesh.creation.axis()
    #     axis.apply_transform(trans)
    #     scene.add_geometry(axis)
    #     # 相机位置小球
    #     sphere = trimesh.creation.uv_sphere(radius=0.05)
    #     sphere.apply_translation(trans[:3,3])
    #     scene.add_geometry(sphere)

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

    # 逐帧显示
    print("按回车显示下一帧，按 Ctrl+C 退出")

    # for i in range(n_frames):
    #     # 复制网格，避免修改原始mesh
    #     mesh_frame = mesh.copy()
    #     # 应用当前帧变换
    #     mesh_frame.apply_transform(pose_sequence[i])
        
    #     # 清除之前的物体，只保留相机和相机点
    #     # trimesh.Scene 没有直接clear接口，故新建scene并重新加相机
    #     # scene = trimesh.Scene()
    #     for cam, trans in camera_to_world.items():
    #         axis = trimesh.creation.axis()
    #         axis.apply_transform(trans)
    #         scene.add_geometry(axis)
    #         sphere = trimesh.creation.uv_sphere(radius=0.05)
    #         sphere.apply_translation(trans[:3,3])
    #         scene.add_geometry(sphere)

    #     # 添加当前帧的网格
    #     scene.add_geometry(mesh_frame)

    #     print(f"显示第 {i+1}/{n_frames} 帧")
    scene.show()

    #     input("按回车继续到下一帧...")

# ==== 示例调用 ====
if __name__ == "__main__":
    camera_params_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    mesh_file = "/home/wys/learning-compliant/crq_ws/data/decim_mesh_files_1/decim_mesh_files/plastic_scoop.obj"
    pose_npy_file = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250624_140312/processed/fd_pose_solver/fd_poses_merged_fixed.npy"

    
    visualize_with_cameras_and_mesh(mesh_file, pose_npy_file, camera_params_file)
