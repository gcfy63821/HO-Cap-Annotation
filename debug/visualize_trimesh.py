import os
import cv2
import numpy as np
from pathlib import Path
import trimesh
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import multiprocessing
import h5py
import torch
from hocap_annotation.layers import MANOGroupLayer, MANOLayer
from hocap_annotation.utils.color_info import *
from hocap_annotation.utils.mano_info import *
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyrender
from pyrender import PerspectiveCamera, DirectionalLight, SpotLight, PointLight, MetallicRoughnessMaterial, Primitive, Mesh, Node, Scene, OffscreenRenderer

# Set up matplotlib for headless rendering
plt.switch_backend('Agg')

def load_pkl_and_get_hand_data(pkl_file):
    # 加载 .pkl 文件
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if 'hand_pose' not in data:
        raise ValueError("No 'hand_pose' found in the .pkl file.")
    hand_pose = data['hand_pose']
    # Extract all relevant fields, using None if not present
    left_hand_pose = np.array(hand_pose.get('left_hand_pose', []))
    left_hand_beta = np.array(hand_pose.get('left_hand_beta', []))
    left_hand_translation = np.array(hand_pose.get('left_hand_translation', []))
    left_hand_base_rot = np.array(hand_pose.get('left_hand_base_rot', []))
    right_hand_pose = np.array(hand_pose.get('right_hand_pose', []))
    right_hand_beta = np.array(hand_pose.get('right_hand_beta', []))
    right_hand_translation = np.array(hand_pose.get('right_hand_translation', []))
    # right_hand_base_rot is not always present
    right_hand_base_rot = np.array(hand_pose.get('right_hand_base_rot', []))
    return {
        'left_hand_pose': left_hand_pose,
        'left_hand_beta': left_hand_beta,
        'left_hand_translation': left_hand_translation,
        'left_hand_base_rot': left_hand_base_rot,
        'right_hand_pose': right_hand_pose,
        'right_hand_beta': right_hand_beta,
        'right_hand_translation': right_hand_translation,
        'right_hand_base_rot': right_hand_base_rot,
    }

def get_betas(b):
    b = np.array(b)
    if b.ndim == 2 and b.shape[0] == 1:
        return b[0]
    return b.squeeze()

def init_mano_layers(hand_data):
    mano_betas_left = get_betas(hand_data['left_hand_beta'])
    mano_betas_right = get_betas(hand_data['right_hand_beta'])
    mano_layer_left = MANOLayer('left', mano_betas_left).to('cuda')
    mano_layer_right = MANOLayer('right', mano_betas_right).to('cuda')
    return mano_layer_left, mano_layer_right

def reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer_right, mano_layer_left, left_pose):
    # Use pose from npy file, other data from pkl
    try:
        pose = torch.tensor(left_pose).to('cuda').unsqueeze(0)
        translation = torch.tensor(hand_data['left_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
        # base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx]).to('cuda') if hand_data['left_hand_base_rot'].ndim == 3 else torch.eye(3).to('cuda')
        base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx]).to('cuda')
        hand_beta = torch.tensor(hand_data['left_hand_beta']).to('cuda')
        
        verts, joints = mano_layer_right(pose, translation)

        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts[:, 0] *= -1
        verts = verts @ base_rot.T
        verts += translation
        faces = mano_layer_left.f.detach().cpu().numpy()
        
        mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
        return mesh
    except Exception as e:
        print(f"Error in reconstruct_left_hand_mesh for frame {frame_idx}: {e}")
        return None

def reconstruct_right_hand_mesh(hand_data, frame_idx, mano_layer_right, right_pose):
    # Use pose from npy file, other data from pkl
    try:
        pose = torch.tensor(right_pose).to('cuda').unsqueeze(0)
        translation = torch.tensor(hand_data['right_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
        
        verts, joints = mano_layer_right(pose, translation)
        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts += translation
        faces = mano_layer_right.f.detach().cpu().numpy()
        
        mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces)
        return mesh
    except Exception as e:
        print(f"Error in reconstruct_right_hand_mesh for frame {frame_idx}: {e}")
        return None

def load_pose(pose_txt):
    with open(pose_txt, 'r') as f:
        arr = np.array([float(x) for x in f.read().strip().split()])
        t = np.array(arr[4:7])
        q = np.array(arr[:4])  # xyzw
        flag = arr[7] if len(arr) > 7 else 0.0  # preserve flag
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T, flag

def load_mano_sequence(mano_file):
    mano_data = np.load(mano_file).astype(np.float32)
    print(f"[INFO] Loaded MANO sequence from {mano_file}, shape: {mano_data.shape}")
    return mano_data  # 返回形状为(2, N, 51)的数组

def load_poses_m(pose_file):
    poses = np.load(pose_file).astype(np.float32)
    mano_sides = ['left','right']
    poses = np.stack(
        [poses[0 if side == "right" else 1] for side in mano_sides], axis=0
    )  # (num_hands, num_frames 51)
    return poses

def load_extrinsics_yaml(yaml_path, serials):
    def create_mat(values):
        return np.array([values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    extr = data["extrinsics"]
    return {s: create_mat(extr[s]) for s in serials}

def read_K_from_yaml(calib_folder, serial, cam_type="color"):
    file_path = Path(calib_folder) / "intrinsics" / f"{serial}.yaml"
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)[cam_type]
    K = np.array([
        [data["fx"], 0.0, data["ppx"]],
        [0.0, data["fy"], data["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K

def render_frame_mesh(frame_idx, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, 
                     orig_vertices, orig_mesh, serials, Ks, extrinsics_dict, base_path, output_path):
    """
    Render hand and tool meshes for a specific frame and camera view
    """
    if frame_idx >= len(pose_data):
        return None
    
    # Load tool pose
    qx, qy, qz, qw, tx, ty, tz = pose_data[frame_idx]
    q = np.array([qx, qy, qz, qw])
    t = np.array([tx, ty, tz])
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    
    # Transform tool mesh
    tool_mesh = orig_mesh.copy()
    tool_mesh.vertices = orig_vertices.copy()
    tool_mesh.apply_transform(T)
    
    # Reconstruct hand meshes
    left_pose = poses_m[0][frame_idx]
    right_pose = poses_m[1][frame_idx]
    
    try:
        left_hand_mesh = reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer_right, mano_layer_left, left_pose)
        right_hand_mesh = reconstruct_right_hand_mesh(hand_data, frame_idx, mano_layer_right, right_pose)
    except Exception as e:
        print(f"Error reconstructing hand meshes for frame {frame_idx}: {e}")
        return None
    
    # Render for each camera view
    for serial_idx, serial in enumerate(serials):
        try:
            # Get camera intrinsics and extrinsics
            K = Ks[serial]
            extrinsics = extrinsics_dict[serial]
            
            # Create pyrender scene
            scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
            
            # Create lights
            direc_l = DirectionalLight(color=[1., 1., 1.], intensity=10.0)
            spot_l = SpotLight(color=[1., 1., 1.], intensity=10.0,
                        innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
            
            # Create camera
            cam = pyrender.IntrinsicsCamera(fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])
            
            # Transform extrinsics for pyrender (flip Y and Z)
            extrs_render = extrinsics @ np.diag([1, -1, -1, 1])
            
            # Add hand meshes with consistent colors (same as visualize_hand_video.py)
            if left_hand_mesh is not None:
                left_hand_mesh.visual.vertex_colors = [0, 102, 204, 255]  # Blue (same as visualize_hand_video.py)
                left_hand_pyrender = Mesh.from_trimesh(left_hand_mesh, smooth=True)
                scene.add(left_hand_pyrender)
            
            if right_hand_mesh is not None:
                right_hand_mesh.visual.vertex_colors = [0, 102, 204, 255]  # Blue (same as visualize_hand_video.py)
                right_hand_pyrender = Mesh.from_trimesh(right_hand_mesh, smooth=True)
                scene.add(right_hand_pyrender)
            
            # Add tool mesh
            tool_mesh.visual.vertex_colors = [128, 128, 128, 255]  # Gray
            tool_pyrender = Mesh.from_trimesh(tool_mesh, smooth=True)
            scene.add(tool_pyrender)
            
            # Add lights and camera
            scene.add(direc_l, pose=extrs_render)
            scene.add(spot_l, pose=extrs_render)
            scene.add(cam, pose=extrs_render)
            
            # Render
            r = OffscreenRenderer(viewport_width=640, viewport_height=480)
            color, depth = r.render(scene)
            r.delete()
            
            # Load original image for overlay
            color_img_path = Path(base_path) / serial / f"color_{frame_idx:06d}.jpg"
            if color_img_path.exists():
                original_img = cv2.imread(str(color_img_path))
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            else:
                original_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # Create mask from depth
            mask = (depth > 0)[:,:,None]
            
            # Overlay rendered mesh on original image
            color_normalized = color.astype(np.float32) / 255.0
            overlay_img = (color_normalized[:, :, :3] * mask + (1 - mask) * original_img.astype(np.float32) / 255.0)
            overlay_img = (overlay_img * 255).astype(np.uint8)
            
            # Save result
            output_file = output_path / f"frame_{frame_idx:06d}_view_{serial}.png"
            
            # Create comparison image
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.title(f'Original - Frame {frame_idx}, View {serial}')
            plt.axis('off')
            plt.imshow(original_img)
            
            plt.subplot(1, 2, 2)
            plt.title(f'Hand+Tool Overlay - Frame {frame_idx}, View {serial}')
            plt.axis('off')
            plt.imshow(overlay_img)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error rendering frame {frame_idx}, view {serial}: {e}")
            continue
    
    return True


def create_videos_for_all_views(output_path, serials, frames_to_process, fps=20, uuid=""):
    """
    Create videos for each camera view from the rendered frames
    """
    try:
        # Create videos directory
        videos_dir = output_path / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        # For each camera view, create a video
        for serial in serials:
            print(f"[INFO] Creating video for camera view {serial}...")
            
            # Get all frame files for this view
            frame_files = sorted(list(output_path.glob(f"frame_*_view_{serial}.png")))
            
            if not frame_files:
                print(f"Warning: No frame files found for view {serial}")
                continue
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            if first_frame is None:
                print(f"Warning: Could not read first frame from {frame_files[0]}")
                continue
            
            height, width, _ = first_frame.shape
            
            # Create video writer
            video_filename = f"mesh_rendering_view_{serial}{uuid}.mp4"
            video_path = videos_dir / video_filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            # Add frames to video
            for frame_file in tqdm(frame_files, desc=f"Creating video for view {serial}"):
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"[INFO] Video saved: {video_path}")
        
        # Also create a combined video showing all views in a grid
        create_combined_grid_video(output_path, serials, frames_to_process, videos_dir, fps, uuid)
        
    except Exception as e:
        print(f"Error creating videos: {e}")

def create_combined_grid_video(output_path, serials, frames_to_process, videos_dir, fps=20, uuid=""):
    """
    Create a combined video showing all camera views in a grid layout
    """
    try:
        print("[INFO] Creating combined grid video...")
        
        # Determine grid layout (2x4 for 8 cameras)
        grid_rows, grid_cols = 2, 4
        if len(serials) <= 4:
            grid_rows, grid_cols = 1, len(serials)
        elif len(serials) <= 6:
            grid_rows, grid_cols = 2, 3
        
        # Get first frame to determine dimensions
        first_frame_files = [output_path / f"frame_{frames_to_process[0]:06d}_view_{serial}.png" for serial in serials]
        first_frames = []
        for frame_file in first_frame_files:
            if frame_file.exists():
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    first_frames.append(frame)
        
        if not first_frames:
            print("Warning: No valid frames found for combined video")
            return
        
        # Get dimensions from first frame
        frame_height, frame_width, _ = first_frames[0].shape
        grid_width = frame_width * grid_cols
        grid_height = frame_height * grid_rows
        
        # Create video writer for combined video
        combined_video_path = videos_dir / f"mesh_rendering_all_views{uuid}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_writer = cv2.VideoWriter(str(combined_video_path), fourcc, fps, (grid_width, grid_height))
        
        # Process each frame
        for frame_idx in tqdm(frames_to_process, desc="Creating combined video"):
            # Create grid frame
            grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Fill grid with frames from each view
            for i, serial in enumerate(serials):
                if i >= grid_rows * grid_cols:
                    break
                
                row = i // grid_cols
                col = i % grid_cols
                
                frame_file = output_path / f"frame_{frame_idx:06d}_view_{serial}.png"
                if frame_file.exists():
                    frame = cv2.imread(str(frame_file))
                    if frame is not None:
                        y_start = row * frame_height
                        y_end = (row + 1) * frame_height
                        x_start = col * frame_width
                        x_end = (col + 1) * frame_width
                        grid_frame[y_start:y_end, x_start:x_end] = frame
            
            combined_writer.write(grid_frame)
        
        combined_writer.release()
        print(f"[INFO] Combined video saved: {combined_video_path}")
        
    except Exception as e:
        print(f"Error creating combined video: {e}")


def process_frame_render(args):
    """
    Process function for multiprocessing
    """
    frame_idx, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, orig_vertices, orig_mesh, serials, Ks, extrinsics_dict, base_path, output_path = args
    return render_frame_mesh(frame_idx, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, 
                           orig_vertices, orig_mesh, serials, Ks, extrinsics_dict, base_path, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_1/20250701_012148", help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为1")
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end_frame", type=int, default=None, help="结束帧")
    parser.add_argument("--camera_indices", type=str, default=None, help="逗号分隔的相机索引列表，如 '0,1,2'，默认全部相机")
    parser.add_argument("--uuid", type=str, default="", help="唯一标识符，用于区分不同运行")
    args = parser.parse_args()

    ################
    data_path = args.data_path
    tool_name = args.tool_name
    pose_file = args.pose_file
    uuid = "_" + args.uuid if args.uuid else ""
    ################

    base_path = f"/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/{data_path}"
    
    # Load tool mesh
    orig_mesh = trimesh.load(f"{base_path}/../../models/{tool_name}/cleaned_mesh_10000.obj", process=False)
    orig_vertices = orig_mesh.vertices.copy()
    
    # Load pose data
    if pose_file == "fd":
        pose_npy_path = f"{base_path}/processed/fd_pose_solver/fd_poses_merged_fixed.npy"
    elif pose_file == "adaptive":
        pose_npy_path = f"{base_path}/processed/fd_pose_solver/adaptive_fd_poses_merged_fixed.npy"
    elif pose_file == "optimized":
        pose_npy_path = f"{base_path}/processed/joint_pose_solver/poses_o.npy"
    
    pose_data = np.load(pose_npy_path)
    print(f"[INFO] Loaded pose data from {pose_npy_path}, shape: {pose_data.shape}")
    
    # Select object based on object_idx
    if pose_data.ndim == 3:
        pose_data = pose_data[args.object_idx - 1]
        print(f"[INFO] Using pose_data[{args.object_idx - 1}], shape: {pose_data.shape}")
    pose_data = pose_data.reshape(-1, 7)
    
    # Load hand data
    pkl_file_path = f"{base_path}/processed/result_hand_optimized.pkl"
    hand_data = load_pkl_and_get_hand_data(pkl_file_path)
    
    # Load MANO pose data
    mano_file = f"{base_path}/processed/joint_pose_solver/poses_m.npy"
    poses_m = load_poses_m(mano_file)
    
    # Initialize MANO layers
    mano_layer_left, mano_layer_right = init_mano_layers(hand_data)
    print(f"[INFO] Loaded mano_betas_left: {get_betas(hand_data['left_hand_beta']).shape}, mano_betas_right: {get_betas(hand_data['right_hand_beta']).shape}")
    print(f"[INFO] Loaded poses_m from {mano_file}, shape: {poses_m.shape}")
    
    # Setup camera data
    serials_all = [f"{i:02d}" for i in range(8)]
    if args.camera_indices is not None:
        camera_indices = [int(idx) for idx in args.camera_indices.split(',')]
        serials = [f"{i:02d}" for i in camera_indices]
    else:
        serials = serials_all
    
    # Load camera intrinsics and extrinsics
    calib_folder = Path("/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration")
    Ks = {s: read_K_from_yaml(calib_folder, s) for s in serials_all}
    extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    extrinsics_dict = load_extrinsics_yaml(extrinsics_yaml, serials_all)
    
    # Create output directory
    output_path = Path(f"debug_output/{data_path}/mesh_rendering{uuid}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine frame range
    num_frames = len(pose_data)
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else num_frames
    frames_to_process = list(range(start_frame, min(end_frame, num_frames)))
    
    print(f"[INFO] Processing {len(frames_to_process)} frames for {len(serials)} camera views")
    
    # Process frames
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=min(4, os.cpu_count()))
    
    args_list = [
        (i, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, orig_vertices, orig_mesh, 
         serials, Ks, extrinsics_dict, base_path, output_path)
        for i in frames_to_process
    ]
    
    results = []
    for result in tqdm(pool.imap(process_frame_render, args_list), total=len(frames_to_process)):
        if result is not None:
            results.append(result)
    
    pool.close()
    pool.join()
    
    # Create videos for each camera view
    print("[INFO] Creating videos for each camera view...")
    create_videos_for_all_views(output_path, serials, frames_to_process, uuid=uuid)
    
    print(f"[INFO] Processing complete. Generated {len(results)} rendered frames.")
    print(f"[INFO] Output saved to {output_path}")
