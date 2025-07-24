import trimesh
import numpy as np
import yaml
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_camera_parameters(camera_params_file):
    """
    Load camera extrinsics from yaml, ignoring tag_0/tag_1.
    Returns: dict {serial: 4x4 np.array}
    """
    with open(camera_params_file, 'r') as f:
        camera_params = yaml.safe_load(f)
    filtered_params = {key: value for key, value in camera_params['extrinsics'].items() if not key.startswith('tag')}
    camera_to_world = {}
    for cam, extrinsics in filtered_params.items():
        mat = np.array(extrinsics).reshape(3, 4)
        transform = np.vstack([mat, [0, 0, 0, 1]])
        camera_to_world[cam] = transform
    return camera_to_world

def load_pose_txt(txt_path):
    """
    Load a 7D pose txt file: [qx, qy, qz, qw, x, y, z] -> 4x4 matrix
    """
    arr = np.loadtxt(txt_path)
    if arr.shape[0] == 7:
        q = arr[:4]
        t = arr[4:7]
    elif arr.shape[0] == 8:  # sometimes flag is saved
        q = arr[:4]
        t = arr[4:7]
    else:
        raise ValueError(f"Unexpected pose txt shape: {arr.shape}")
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def preprocess_flip_z_to_match_ref(tool_poses_world, ref_cam='07'):
    """
    For each cam's pose in world, flip z axis if needed so that its z direction is closer to ref_cam's z direction.
    Returns a new dict with possibly flipped poses.
    """
    from scipy.spatial.transform import Rotation as R
    processed = {}
    print(tool_poses_world.keys())
    ref_pose = tool_poses_world[ref_cam]
    ref_z = ref_pose[:3, 2]
    ref_rot = ref_pose[:3, :3]
    for cam, pose in tool_poses_world.items():
        if cam == ref_cam:
            processed[cam] = pose
            continue
        R0 = pose[:3, :3]
        t0 = pose[:3, 3]
        z0 = R0[:, 2]
        # Mirrored rotation: flip z axis
        R_flip = R0.copy()
        R_flip[:, 2] *= -1
        # To keep right-handed, also flip x or y (here flip x)
        R_flip[:, 0] *= -1
        # Compare angle between z0 and ref_z
        angle_orig = np.arccos(np.clip(np.dot(z0, ref_z) / (np.linalg.norm(z0) * np.linalg.norm(ref_z)), -1, 1))
        z_flip = R_flip[:, 2]
        angle_flip = np.arccos(np.clip(np.dot(z_flip, ref_z) / (np.linalg.norm(z_flip) * np.linalg.norm(ref_z)), -1, 1))
        if angle_flip < angle_orig:
            T_new = np.eye(4)
            T_new[:3, :3] = R_flip
            T_new[:3, 3] = t0
            processed[cam] = T_new
            print(f"[INFO] Cam {cam}: flipped z axis to match cam {ref_cam} (angle_orig={np.rad2deg(angle_orig):.2f} deg, angle_flip={np.rad2deg(angle_flip):.2f} deg)")
        else:
            processed[cam] = pose
    return processed

def visualize_tracking_pose(
    ob_in_cam_dir,  # directory containing per-cam/000000.txt
    extrinsics_yaml,  # extrinsics yaml file
    serials=None,
    frame_idx=210
):
    # Load extrinsics
    camera_to_world = load_camera_parameters(extrinsics_yaml)
    if serials is None:
        serials = sorted(camera_to_world.keys())
    # Load tracking result for each camera
    tool_poses_world = {}
    for cam in serials:
        pose_txt = Path(ob_in_cam_dir) / cam / f"{frame_idx:06d}.txt"
        if not pose_txt.exists():
            print(f"[WARN] {pose_txt} not found, skipping.")
            continue
        ob_in_cam = load_pose_txt(pose_txt)
        cam2world = camera_to_world[cam]
        ob_in_world = cam2world @ ob_in_cam
        tool_poses_world[cam] = ob_in_world
    # Preprocess: flip z axis if needed to match cam7
    tool_poses_world = preprocess_flip_z_to_match_ref(tool_poses_world, ref_cam='07')
    # Create scene
    scene = trimesh.Scene()
    # Add camera axes
    for cam in serials:
        cam2world = camera_to_world[cam]
        cam_axis = trimesh.creation.axis(origin_size=0.01, axis_length=0.05)
        cam_axis.apply_transform(cam2world)
        scene.add_geometry(cam_axis, node_name=f"cam_{cam}")
    # Add tool axes (tracking result in world)
    for cam, ob_in_world in tool_poses_world.items():
        tool_axis = trimesh.creation.axis(origin_size=0.01, axis_length=0.08)
        tool_axis.apply_transform(ob_in_world)
        scene.add_geometry(tool_axis, node_name=f"tool_{cam}")
        # Add cam number as text label next to tool axis
        try:
            text_path = trimesh.path.creation.text(cam, height=0.025)
            # Place text at tool axis origin, offset a bit in +z
            text_tf = np.eye(4)
            text_tf[:3, 3] = ob_in_world[:3, 3] + ob_in_world[:3, 2] * 0.04
            text_path.apply_transform(text_tf)
            scene.add_geometry(text_path, node_name=f"label_{cam}")
        except Exception as e:
            print(f"[WARN] Could not create text for cam {cam}: {e}")
    # Show scene
    scene.show()

def plot_pose_axes(ax, T, label, color='k', axis_length=0.08):
    """Plot a coordinate frame at T (4x4), with a label."""
    origin = T[:3, 3]
    R = T[:3, :3]
    # X, Y, Z axes
    axes = np.eye(3) * axis_length
    for i, c in zip(range(3), ['r', 'g', 'b']):
        ax.quiver(
            origin[0], origin[1], origin[2],
            R[0, i]*axis_length, R[1, i]*axis_length, R[2, i]*axis_length,
            color=c, linewidth=2
        )
    # Add label
    ax.text(
        origin[0], origin[1], origin[2] + axis_length * 1.1,
        label, color=color, fontsize=10, weight='bold'
    )

def visualize_tracking_pose_matplotlib(
    ob_in_cam_dir,  # directory containing per-cam/000000.txt
    extrinsics_yaml,  # extrinsics yaml file
    serials=None,
    frame_idx=210
):
    # Load extrinsics
    camera_to_world = load_camera_parameters(extrinsics_yaml)
    if serials is None:
        serials = sorted(camera_to_world.keys())
    # Load tracking result for each camera
    tool_poses_world = {}
    for cam in serials:
        pose_txt = Path(ob_in_cam_dir) / cam / f"{frame_idx:06d}.txt"
        if not pose_txt.exists():
            print(f"[WARN] {pose_txt} not found, skipping.")
            continue
        ob_in_cam = load_pose_txt(pose_txt)
        cam2world = camera_to_world[cam]
        ob_in_world = cam2world @ ob_in_cam
        tool_poses_world[cam] = ob_in_world
    # Preprocess: flip z axis if needed to match cam7
    tool_poses_world = preprocess_flip_z_to_match_ref(tool_poses_world, ref_cam='07')
    # Matplotlib 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot each tool pose
    for cam, T in tool_poses_world.items():
        plot_pose_axes(ax, T, label=f"Cam {cam}", color='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Tool poses in world frame (frame {frame_idx})")
    ax.set_box_aspect([1,1,1])
    plt.show()

if __name__ == "__main__":
    ob_in_cam_dir = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/videos_0713/wooden_brush_1_1/processed/fd_pose_solver/wooden_brush/ob_in_cam"
    extrinsics_yaml = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/calibration/extrinsics/extrinsics.yaml"
    serials = [f"{i:02d}" for i in range(8)]
    frame_idx = 0
    visualize_tracking_pose(ob_in_cam_dir, extrinsics_yaml, serials, frame_idx=frame_idx)
    # Also show matplotlib visualization
    visualize_tracking_pose_matplotlib(ob_in_cam_dir, extrinsics_yaml, serials, frame_idx=frame_idx)
