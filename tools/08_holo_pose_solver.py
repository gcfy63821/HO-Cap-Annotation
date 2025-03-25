import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from hocap_annotation.utils import *
from hocap_annotation.loaders import HOCapLoader
from hocap_annotation.layers import MANOGroupLayer
from hocap_annotation.wrappers.foundationpose import *

cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
set_logging_format()


class HoloRenderer:
    def __init__(self, width, height, znear=0.01, zfar=10.0) -> None:
        self._znear = znear
        self._zfar = zfar
        self._width = width
        self._height = height
        self._cam_node = None
        self._obj_nodes = []
        self._seg_node_map = {}

    def add_camera(self, cam_K, name):
        """Add a camera to the scene with its intrinsic matrix `cam_K`."""
        self._cam_node = pyrender.Node(
            name=name,
            camera=pyrender.IntrinsicsCamera(
                fx=cam_K[0, 0],
                fy=cam_K[1, 1],
                cx=cam_K[0, 2],
                cy=cam_K[1, 2],
                znear=self._znear,
                zfar=self._zfar,
            ),
        )

    def add_object_mesh(self, mesh, name):
        """Add a mesh to the scene."""
        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh),
        )
        self._obj_nodes.append(node)
        self._seg_node_map[node] = (len(self._obj_nodes), 0, 0)

    def _add_node(self, scene, node, parent_node, pose):
        scene.add_node(node, parent_node=parent_node)
        scene.set_pose(node, pose=pose)

    def _render_scene(self, cam_pose, mano_mesh, object_poses, render_flags):
        scene = pyrender.Scene(
            bg_color=[0, 0, 0, 1], ambient_light=[1.0, 1.0, 1.0, 1.0]
        )

        r = pyrender.OffscreenRenderer(self._width, self._height)

        try:
            # Add world node
            world_node = scene.add_node(pyrender.Node(name="world"))

            # Add camera
            self._add_node(
                scene,
                self._cam_node,
                parent_node=world_node,
                pose=cam_pose @ cvcam_in_glcam,
            )

            # Add MANO mesh
            mano_node = pyrender.Node(
                name="mano", mesh=pyrender.Mesh.from_trimesh(mano_mesh)
            )
            scene.add_node(mano_node, parent_node=world_node)

            # Add object meshes
            for node, pose in zip(self._obj_nodes, object_poses):
                self._add_node(scene, node, parent_node=world_node, pose=pose)

            seg_node_map = self._seg_node_map.copy()
            seg_node_map[mano_node] = (0, 0, 0)

            color, depth = r.render(scene, render_flags, seg_node_map)
        finally:
            r.delete()

        return color, depth

    def get_render_color(self, cam_pose, mano_mesh, object_poses):
        color, _ = self._render_scene(cam_pose, mano_mesh, object_poses, 0)
        return color

    def get_render_seg(self, cam_pose, mano_mesh, object_poses):
        seg, _ = self._render_scene(
            cam_pose, mano_mesh, object_poses, pyrender.RenderFlags.SEG
        )
        seg = seg[..., 0]
        return seg


def draw_texts_on_image(
    image,
    texts,
    position=(10, 30),
    font_scale=1.0,
    font_thickness=2,
    font_color=(0, 255, 0),
):
    """
    Draw a list of text strings on the top-left corner of an image.

    Args:
        image (np.ndarray): The input image.
        texts (list of str): The list of texts to be drawn.
        position (tuple): Top-left position of the first text.
        font_scale (float): Font size multiplier for the text.
        font_thickness (int): Thickness of the text.
        font_color (tuple): Color of the text in (B, G, R) format.
    """
    x, y = position
    line_height = int(
        cv2.getTextSize(texts[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[
            0
        ][1]
        * 1.5
    )

    for i, text in enumerate(texts):
        y_position = y + i * line_height
        cv2.putText(
            image,
            text,
            (x, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def calculate_final_score(
    iou_score: float,
    dice_score: float,
    rgb_diff: float,
    w_iou: float = 1.0,
    w_dice: float = 1.0,
    w_rgb: float = 0.5,
) -> float:
    """
    Calculate the final score based on IoU, Dice, and RGB difference.

    Args:
        iou_score (float): The IoU score (between 0 and 1).
        dice_score (float): The Dice coefficient score (between 0 and 1).
        rgb_diff (float): The RGB difference score (lower is better).
        max_rgb_diff (float): The maximum RGB difference for normalization.
        w_iou (float, optional): Weight for IoU score. Defaults to 0.5.
        w_dice (float, optional): Weight for Dice coefficient. Defaults to 0.5.
        w_rgb (float, optional): Weight for RGB difference. Defaults to 0.5.

    Returns:
        float: The final score to select the best pose (higher is better).
    """
    # Calculate the final score (higher is better)
    final_score = (
        w_iou * iou_score  # Contribution from IoU
        + w_dice * dice_score  # Contribution from Dice coefficient
        # - w_rgb * rgb_diff  # Inverted contribution from RGB difference
    )

    return final_score


def get_debug_info(
    frame_id, rgb_image, mask_image, renderer, mano_mesh, cam_pose, object_poses
):
    masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_image.astype(np.uint8))
    color = renderer.get_render_color(cam_pose, mano_mesh, object_poses)
    mask = renderer.get_render_seg(cam_pose, mano_mesh, object_poses)
    mask_overlay = draw_object_mask_overlay(rgb_image, mask, 0.7)
    color_overlay = draw_image_overlay(rgb_image, color, 0.7)
    mask_dice_score = get_mask_dice_coefficient(mask_image, mask)
    mask_iou_score = get_mask_iou(mask_image, mask)
    color_diff_score = get_rgb_difference(masked_rgb, color)
    final_score = calculate_final_score(
        mask_iou_score, mask_dice_score, color_diff_score
    )
    draw_texts_on_image(
        color_overlay,
        texts=[
            f"frame_id: {frame_id}",
            f"dice_score: {mask_dice_score:.4f}",
            f"iou_score: {mask_iou_score:.4f}",
            f"color_diff: {color_diff_score:.4f}",
            f"final_score: {final_score:.4f}",
        ],
    )
    return (
        color,
        mask,
        color_overlay,
        mask_overlay,
        mask_dice_score,
        mask_iou_score,
        color_diff_score,
        final_score,
    )


def run_fd_pose_estimator(sequence_folder, refine_iter):
    t_s = time.time()

    set_seed(0)

    sequence_folder = Path(sequence_folder)
    # Make folders
    save_folder = sequence_folder / "processed/holo_pose_solver"
    save_folder.mkdir(parents=True, exist_ok=True)
    save_data_folder = save_folder / "optim_data"
    save_data_folder.mkdir(parents=True, exist_ok=True)
    save_vis_folder = save_folder / "vis_best_frame"
    save_vis_folder.mkdir(parents=True, exist_ok=True)
    debug_dir = save_folder / "debug"

    # Load parameters from reader
    masks_folder = sequence_folder / "processed/segmentation/sam2"
    reader = HOCapLoader(sequence_folder)
    num_frames = reader.num_frames
    hl_serial = reader.hl_serial
    cam_K = reader.hl_K
    pv_width = reader.hl_width
    pv_height = reader.hl_height
    object_ids = reader.object_ids
    num_objects = len(object_ids)
    mano_sides = reader.mano_sides
    mano_beta = reader.mano_beta
    hl_pv_poses = quat_to_mat(
        np.load(sequence_folder / "holo_pv_poses_raw.npy").astype(np.float32)
    )
    object_meshes = [
        trimesh.load_mesh(str(m_file), process=False)
        for m_file in reader.object_textured_files
    ]

    poses_o = np.load(sequence_folder / "processed/joint_pose_solver/poses_o.npy")
    poses_m = np.load(sequence_folder / "processed/joint_pose_solver/poses_m.npy")
    poses_m = np.concatenate(
        [poses_m[0 if s == "right" else 1] for s in mano_sides], axis=1
    )
    poses_m = torch.from_numpy(poses_m).to(CFG.device)
    mano_group_layer = MANOGroupLayer(mano_sides, [mano_beta for _ in mano_sides]).to(
        CFG.device
    )
    verts_m, joints_m = mano_group_layer(poses_m)
    verts_m = verts_m.cpu().numpy()
    faces_m = [mano_group_layer.f.cpu().numpy()]
    colors_m = []
    for i, side in enumerate(mano_sides):
        faces_m.append(np.array(NEW_MANO_FACES[side]) + i * NUM_MANO_VERTS)
        colors_m.append(
            [HAND_COLORS[1 if side == "right" else 2].rgb_norm] * NUM_MANO_VERTS
        )
    faces_m = np.concatenate(faces_m, axis=0).astype(np.int64)
    colors_m = np.concatenate(colors_m, axis=0).astype(np.float32)

    # Initialize renderer
    renderer = HoloRenderer(pv_width, pv_height)
    renderer.add_camera(cam_K, "camera")
    for i, m in enumerate(object_meshes):
        renderer.add_object_mesh(m, object_ids[i])

    # Initialize pose estimator
    box_mesh = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    object_mesh = trimesh.Trimesh(
        vertices=box_mesh.vertices.copy(), faces=box_mesh.faces.copy()
    )
    est = FoundationPose(
        model_pts=object_mesh.vertices,
        model_normals=object_mesh.vertex_normals,
        mesh=object_mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug_dir=debug_dir,
        debug=1,
    )

    # best_frame_ids = []
    all_optim_data = []
    best_pv_poses = []

    for frame_id in range(num_frames):
        frame_range = list(range(max(0, frame_id - 5), min(num_frames, frame_id + 1)))
        object_poses = [quat_to_mat(poses_o[i, frame_id]) for i in range(num_objects)]

        mano_mesh = trimesh.Trimesh(
            vertices=verts_m[frame_id], faces=faces_m, vertex_colors=colors_m
        )
        mano_mesh.vertex_normals

        object_mesh = trimesh.util.concatenate(
            [
                object_meshes[i].copy().apply_transform(p)
                for i, p in enumerate(object_poses)
            ]
        )

        color = reader.get_color(hl_serial, frame_id)
        mask = read_mask_image(
            masks_folder / "hololens_kv5h72" / "mask" / f"mask_{frame_id:06d}.png"
        ).astype(bool)
        depth = np.zeros((pv_height, pv_width), dtype=np.float32)
        rgb = (
            color.copy()
            if mask.sum() == 0
            else cv2.bitwise_and(color, color, mask=mask.astype(np.uint8))
        )

        # Update mesh for pose estimator
        est.reset_object(
            model_pts=object_mesh.vertices,
            model_normals=object_mesh.vertex_normals,
            mesh=object_mesh,
        )

        optim_data = {}
        final_scores = []
        render_images = []
        optim_poses = []

        for f_id in frame_range:
            hl_pv_pose = hl_pv_poses[f_id]
            ob_in_cam_0 = np.linalg.inv(hl_pv_pose)

            (
                r_color,
                r_mask,
                r_color_overlay,
                r_mask_overlay,
                r_mask_dice_score,
                r_mask_iou_score,
                r_color_diff_score,
                r_final_score,
            ) = get_debug_info(
                f_id, color, mask, renderer, mano_mesh, hl_pv_pose, object_poses
            )

            # Run pose estimation
            ob_in_cam = est.track_one(
                rgb=rgb,
                depth=depth,
                K=cam_K,
                iteration=refine_iter,
                prev_pose=ob_in_cam_0,
            )

            hl_pv_pose_refined = np.linalg.inv(ob_in_cam)

            (
                r_color_refined,
                r_mask_refined,
                r_color_overlay_refined,
                r_mask_overlay_refined,
                r_mask_dice_score_refined,
                r_mask_iou_score_refined,
                r_color_diff_score_refined,
                r_final_score_refined,
            ) = get_debug_info(
                f_id, color, mask, renderer, mano_mesh, hl_pv_pose_refined, object_poses
            )

            # Save results
            optim_data[f_id] = {
                "hl_pv_pose": hl_pv_pose.copy(),
                "hl_pv_pose_refined": hl_pv_pose_refined.copy(),
                "r_color_diff_score": r_color_diff_score,
                "r_mask_dice_score": r_mask_dice_score,
                "r_mask_iou_score": r_mask_iou_score,
                "r_color_diff_score_refined": r_color_diff_score_refined,
                "r_mask_dice_score_refined": r_mask_dice_score_refined,
                "r_mask_iou_score_refined": r_mask_iou_score_refined,
                "r_final_score": r_final_score,
                "r_final_score_refined": r_final_score_refined,
            }

            final_scores.append(r_final_score)
            final_scores.append(r_final_score_refined)

            render_images.append(r_color_overlay)
            render_images.append(r_color_overlay_refined)

        # Save results
        best_score_idx = np.argmax(final_scores)
        best_frame_idx = best_score_idx // 2
        best_pose_idx = best_score_idx % 2
        best_frame_id = frame_range[best_frame_idx]
        best_pose = optim_data[best_frame_id][
            "hl_pv_pose" if best_pose_idx == 0 else "hl_pv_pose_refined"
        ]
        best_frame_vis = render_images[best_score_idx]
        optim_data["best_frame_id"] = best_frame_id
        optim_data["best_pose"] = best_pose
        best_pv_poses.append(best_pose)

        all_optim_data.append(optim_data)
        # best_frame_ids.append(best_frame_id)

        write_data_to_pickle(save_data_folder / f"{frame_id:06d}.pkl", optim_data)
        write_rgb_image(save_vis_folder / f"{frame_id:06d}.png", best_frame_vis)

    # np.savetxt(save_folder / "best_frame_ids.txt", best_frame_ids, fmt="%06d")
    best_pv_poses = np.stack(best_pv_poses, axis=0).astype(np.float32)
    np.save(save_folder / "poses_pv.npy", best_pv_poses)

    # Save best frames to video
    logging.info("Creating video from best frames...")
    all_best_frame_files = sorted(save_vis_folder.glob("*.png"))
    create_video_from_image_files(
        save_folder / "vis_holo_pose_solver.mp4", all_best_frame_files, preload=True
    )

    t_e = time.time()

    logging.info(f"done!!! time: {t_e - t_s:.3f}s")


def args_parser():

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hololens pose solver")
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--refine_iter",
        type=int,
        default=5,
        help="Number of iterations for pose refinement.",
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide a valid sequence folder path.")

    run_fd_pose_estimator(args.sequence_folder, args.refine_iter)
