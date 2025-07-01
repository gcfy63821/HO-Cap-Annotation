import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端（适合保存图像）

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm

def draw_cube(ax, T, size=0.05):
    """Draw a cube transformed by 4x4 pose matrix T."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Define 8 cube corners
    s = size / 2.0
    corners = np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ])
    # Transform to world
    corners_homo = np.hstack([corners, np.ones((8, 1))])  # (8,4)
    transformed = (T @ corners_homo.T).T[:, :3]

    # Faces
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]
    poly3d = [[transformed[i] for i in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='lightblue', linewidths=1, edgecolors='k', alpha=0.8))

def pose7d_to_matrix(pose7d):
    """Convert 7D [x, y, z, qx, qy, qz, qw] to 4x4 SE3 matrix."""
    t = pose7d[:3]
    quat = pose7d[3:]
    R_mat = R.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T

def visualize_pose_sequence(npy_path, save_video_path=None, show=True):
    poses = np.load(npy_path)  # (1, N, 7)
    if poses.ndim == 3 and poses.shape[0] == 1:
        poses = poses[0]  # (N, 7)
    N = poses.shape[0]
    print(f"Loaded {N} poses")

    frames = []
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111, projection='3d')
    fig = plt.figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)  # 绑定 canvas
    ax = fig.add_subplot(111, projection='3d')

    # Get overall motion bounds to fix view
    all_trans = poses[:, :3]
    xlim = (all_trans[:, 0].min() - 0.1, all_trans[:, 0].max() + 0.1)
    ylim = (all_trans[:, 1].min() - 0.1, all_trans[:, 1].max() + 0.1)
    zlim = (all_trans[:, 2].min() - 0.1, all_trans[:, 2].max() + 0.1)

    for i in tqdm(range(N)):
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')
        ax.view_init(elev=30, azim=45)

        pose = poses[i]
        if np.all(pose == -1):
            ax.text2D(0.3, 0.5, "Invalid Pose", transform=ax.transAxes, color='red')
        else:
            T = pose7d_to_matrix(pose)
            draw_cube(ax, T, size=0.05)

        # fig.canvas.draw()
        # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()  # 去掉 alpha 通道

        # img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        frames.append(img)

    plt.close()

    # if save_video_path:
    #     print(f"Saving video to: {save_video_path}")
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     h, w, _ = frames[0].shape
    #     video = cv2.VideoWriter(save_video_path, fourcc, 20, (w, h))
    #     for frame in frames:
    #         video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     video.release()

    if save_video_path:
        print(f"Saving video to: {save_video_path}")
        h, w, _ = frames[0].shape
        # 确保 w 和 h 是偶数，避免编码器不兼容问题
        w = w - w % 2
        h = h - h % 2
        size = (w, h)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 替换掉 avc1

        video = cv2.VideoWriter(save_video_path, fourcc, 20, size)

        for frame in frames:
            resized = cv2.resize(frame, size)
            video.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        video.release()
        print("Video saved.")


    if show:
        for f in frames:
            cv2.imshow("Pose Visualization", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

# ==== Example usage ====
if __name__ == "__main__":
    visualize_pose_sequence(
        npy_path="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250624_140312/processed/fd_pose_solver/fd_poses_merged_fixed.npy",
        save_video_path="pose_tracking_vis.mp4",
        show=False  # 设置为 True 会逐帧播放
    )
