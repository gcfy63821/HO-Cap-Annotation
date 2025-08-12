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
from scipy.spatial.distance import cdist

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

def reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer, left_pose):
    # Use pose from npy file, other data from pkl
    try:
        pose = torch.tensor(left_pose).to('cuda').unsqueeze(0)
        translation = torch.tensor(hand_data['left_hand_translation'][frame_idx]).to('cuda').unsqueeze(0)
        base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx]).to('cuda') if hand_data['left_hand_base_rot'].ndim == 3 else torch.eye(3).to('cuda')
        hand_beta = torch.tensor(hand_data['left_hand_beta']).to('cuda')
        
        verts, joints = mano_layer(pose, translation)

        if verts.size(0) == 1:
            verts = verts.squeeze(0)
            joints = joints.squeeze(0)
        
        root_trans = joints[0].clone().detach()
        verts -= root_trans
        verts[:, 0] *= -1
        verts = verts @ base_rot.T
        verts += translation
        faces = mano_layer.f.detach().cpu().numpy()
        
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

def compute_distance_heatmap(hand_mesh, tool_mesh, max_distance=0.05):
    """
    Compute distance heatmap between hand and tool meshes
    Returns distance values for hand vertices
    """
    if hand_mesh is None or tool_mesh is None:
        return None
    
    # Get hand vertices
    hand_vertices = hand_mesh.vertices
    
    # Get tool vertices
    tool_vertices = tool_mesh.vertices
    
    # Compute distances from hand vertices to tool vertices
    distances = cdist(hand_vertices, tool_vertices)
    
    # For each hand vertex, get the minimum distance to any tool vertex
    min_distances = np.min(distances, axis=1)
    
    # Clip distances to max_distance
    min_distances = np.clip(min_distances, 0, max_distance)
    
    return min_distances

def create_contact_visualization(hand_mesh, tool_mesh, distances, output_path, frame_idx):
    """
    Create a 3D visualization showing hand, tool, and contact heatmap
    """
    if hand_mesh is None or tool_mesh is None or distances is None:
        return
    
    output_file = output_path / f"contact_heatmap_{frame_idx:06d}.png"
    
    # Try 3D rendering first
    try:
        # Create a new scene
        scene = trimesh.Scene()
        
        # Add tool mesh (gray)
        tool_mesh.visual.face_colors = [128, 128, 128, 255]  # Gray
        scene.add_geometry(tool_mesh)
        
        # Create hand mesh with distance-based colors
        hand_mesh_copy = hand_mesh.copy()
        
        # Normalize distances for color mapping (0=red/contact, 1=blue/far)
        normalized_distances = distances / np.max(distances) if np.max(distances) > 0 else distances
        
        # Create color map: red (close) to blue (far)
        colors = np.zeros((len(hand_mesh_copy.vertices), 4), dtype=np.uint8)
        colors[:, 0] = (normalized_distances * 255).astype(np.uint8)  # Red channel (close = red)
        colors[:, 2] = ((1 - normalized_distances) * 255).astype(np.uint8)  # Blue channel (far = blue)
        colors[:, 3] = 255  # Alpha
        
        # Apply colors to vertices
        hand_mesh_copy.visual.vertex_colors = colors
        
        scene.add_geometry(hand_mesh_copy)
        
        # Set camera position for better view
        scene.camera.resolution = [800, 600]
        scene.camera.fov = [60, 60]
        
        # Try to set a good camera angle
        try:
            # Get bounding box of both meshes
            all_vertices = np.vstack([hand_mesh.vertices, tool_mesh.vertices])
            center = np.mean(all_vertices, axis=0)
            extent = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
            max_extent = np.max(extent)
            
            # Position camera
            camera_distance = max_extent * 2
            scene.camera_transform = trimesh.transformations.look_at(
                points=center,
                eye=center + [camera_distance, camera_distance, camera_distance],
                up=[0, 0, 1]
            )
        except:
            pass
        
        # Try headless rendering
        png = scene.save_image(resolution=[800, 600], visible=False)
        with open(output_file, 'wb') as f:
            f.write(png)
        return output_file
        
    except Exception as e:
        print(f"Warning: 3D rendering failed for frame {frame_idx}: {e}")
        # Fallback to 2D distance plot
        create_distance_plot(distances, output_file, frame_idx)
        return output_file

def create_distance_point_cloud_video(hand_mesh, tool_mesh, distances, output_path, frame_idx, max_distance=0.05):
    """
    Create a point cloud visualization of hand vertices colored by distance to tool
    Red = close, Green = far
    """
    if hand_mesh is None or tool_mesh is None or distances is None:
        return None
    
    # Get hand vertices
    hand_vertices = hand_mesh.vertices
    
    # Normalize distances for color mapping (0=red/close, 1=green/far)
    normalized_distances = np.clip(distances / max_distance, 0, 1)
    
    # Create color map: red (close) to green (far)
    colors = np.zeros((len(hand_vertices), 3), dtype=np.uint8)
    colors[:, 0] = ((1 - normalized_distances) * 255).astype(np.uint8)  # Red channel (close = red)
    colors[:, 1] = (normalized_distances * 255).astype(np.uint8)  # Green channel (far = green)
    colors[:, 2] = 0  # Blue channel (always 0)
    
    # Create a simple 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hand vertices colored by distance
    scatter = ax.scatter(hand_vertices[:, 0], hand_vertices[:, 1], hand_vertices[:, 2], 
                        c=colors/255.0, s=1, alpha=0.8)
    
    # Plot tool mesh as wireframe
    if len(tool_mesh.vertices) > 0:
        ax.plot_trisurf(tool_mesh.vertices[:, 0], tool_mesh.vertices[:, 1], tool_mesh.vertices[:, 2],
                       triangles=tool_mesh.faces, alpha=0.3, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Hand-Tool Distance Heatmap - Frame {frame_idx}\nRed=Close, Green=Far')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the frame
    output_file = output_path / f"contact_video_frame_{frame_idx:06d}.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_distance_plot(distances, output_file, frame_idx):
    """
    Create a simple 2D plot of distance distribution as fallback
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Number of vertices')
        ax1.set_title(f'Distance Distribution - Frame {frame_idx}')
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f"""
        Min: {np.min(distances):.4f}m
        Max: {np.max(distances):.4f}m
        Mean: {np.mean(distances):.4f}m
        Std: {np.std(distances):.4f}m
        Contact vertices: {(distances < 0.01).sum()}
        """
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_title('Distance Statistics')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create distance plot for frame {frame_idx}: {e}")
        # Last resort: save raw distance data
        np.save(output_file.with_suffix('.npy'), distances)

def process_frame_contact(args):
    frame_idx, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, orig_vertices, orig_mesh, output_path, max_distance = args
    
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
        left_hand_mesh = reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer_left, left_pose)
        right_hand_mesh = reconstruct_right_hand_mesh(hand_data, frame_idx, mano_layer_right, right_pose)
    except Exception as e:
        print(f"Error reconstructing hand meshes for frame {frame_idx}: {e}")
        return None
    
    # Compute contact heatmaps
    left_distances = compute_distance_heatmap(left_hand_mesh, tool_mesh, max_distance)
    right_distances = compute_distance_heatmap(right_hand_mesh, tool_mesh, max_distance)
    
    # Create video frames
    left_output = None
    right_output = None
    
    if left_distances is not None:
        left_output = create_distance_point_cloud_video(left_hand_mesh, tool_mesh, left_distances, output_path / "left_hand", frame_idx, max_distance)
    
    if right_distances is not None:
        right_output = create_distance_point_cloud_video(right_hand_mesh, tool_mesh, right_distances, output_path / "right_hand", frame_idx, max_distance)
    
    # Save distance data
    if left_distances is not None:
        np.save(output_path / "left_hand" / f"distances_{frame_idx:06d}.npy", left_distances)
    
    if right_distances is not None:
        np.save(output_path / "right_hand" / f"distances_{frame_idx:06d}.npy", right_distances)
    
    return {
        'frame_idx': frame_idx,
        'left_distances': left_distances,
        'right_distances': right_distances,
        'left_output': left_output,
        'right_output': right_output
    }

def create_video_from_frames(frames_dir, output_filename, fps=20):
    """
    Create a video from PNG frames in the given directory
    """
    try:
        # Get all frame files
        frame_files = sorted(list(frames_dir.glob("contact_video_frame_*.png")))
        
        if not frame_files:
            print(f"Warning: No frame files found in {frames_dir}")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            print(f"Warning: Could not read first frame from {frame_files[0]}")
            return
        
        height, width, _ = first_frame.shape
        
        # Create video writer
        output_path = frames_dir / output_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_file in tqdm(frame_files, desc=f"Creating {output_filename}"):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        print(f"[INFO] Video saved: {output_path}")
        
    except Exception as e:
        print(f"Error creating video {output_filename}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_1/20250701_012148", help="数据路径，如 test_1/20250701_012148")
    parser.add_argument("--tool_name", type=str, default="blue_scooper", help="工具名，如 blue_scooper")
    parser.add_argument("--pose_file", type=str, default="fd", choices=["fd", "adaptive", "optimized"], help="选择foundation pose 或 optimized")
    parser.add_argument("--object_idx", type=int, default=1, help="物体索引，默认为1")
    parser.add_argument("--max_distance", type=float, default=0.05, help="最大距离阈值（米）")
    args = parser.parse_args()

    ################
    data_path = args.data_path
    tool_name = args.tool_name
    pose_file = args.pose_file
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
    
    # Create output directories
    output_path = Path(f"debug_output/{data_path}/contact_heatmap")
    (output_path / "left_hand").mkdir(parents=True, exist_ok=True)
    (output_path / "right_hand").mkdir(parents=True, exist_ok=True)
    
    # Get number of frames
    num_frames = len(pose_data)
    print(f"[INFO] Processing {num_frames} frames")
    
    # Process frames
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=min(4, os.cpu_count()))
    
    args_list = [
        (i, pose_data, hand_data, mano_layer_left, mano_layer_right, poses_m, orig_vertices, orig_mesh, output_path, args.max_distance)
        for i in range(num_frames)
    ]
    
    results = []
    for result in tqdm(pool.imap(process_frame_contact, args_list), total=num_frames):
        if result is not None:
            results.append(result)
    
    pool.close()
    pool.join()
    
    # Create videos from the frames
    print("[INFO] Creating videos from frames...")
    create_video_from_frames(output_path / "left_hand", "left_hand_contact_video.mp4", fps=20)
    create_video_from_frames(output_path / "right_hand", "right_hand_contact_video.mp4", fps=20)
    
    # Create summary statistics
    print(f"[INFO] Processing complete. Generated {len(results)} contact heatmaps.")
    print(f"[INFO] Output saved to {output_path}")
    
    # Optional: Create a summary video or statistics
    if results:
        all_left_distances = []
        all_right_distances = []
        
        for result in results:
            if result['left_distances'] is not None:
                all_left_distances.extend(result['left_distances'])
            if result['right_distances'] is not None:
                all_right_distances.extend(result['right_distances'])
        
        if all_left_distances:
            print(f"[STATS] Left hand - Min distance: {np.min(all_left_distances):.4f}m, Max distance: {np.max(all_left_distances):.4f}m, Mean distance: {np.mean(all_left_distances):.4f}m")
        
        if all_right_distances:
            print(f"[STATS] Right hand - Min distance: {np.min(all_right_distances):.4f}m, Max distance: {np.max(all_right_distances):.4f}m, Mean distance: {np.mean(all_right_distances):.4f}m")
