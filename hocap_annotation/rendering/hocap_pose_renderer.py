from ..utils import *
from ..loaders import HOCapLoader
from .offscreen_renderer import OffscreenRenderer


class HOCapPoseRenderer:
    def __init__(self, sequence_folder, debug=False, log_file=None) -> None:
        self._logger = get_logger(
            self.__class__.__name__,
            log_level="DEBUG" if debug else "INFO",
            log_file=log_file,
        )
        self._data_folder = Path(sequence_folder)
        self._reader = HOCapLoader(sequence_folder)
        self._num_frames = self._reader.num_frames
        self._rs_width = self._reader.rs_width
        self._rs_height = self._reader.rs_height
        self._rs_serials = self._reader.rs_serials
        self._valid_serials = self._reader.get_valid_seg_serials()
        self._Ks = self._reader.rs_Ks
        self._object_ids = self._reader.object_ids
        self._camera_poses = self._reader.extr2world
        self._poses_dict = {
                "ob_poses_in_world": None,
                "ob_poses_in_cam": None,
                "mano_verts": None,
                "mano_faces": None,
                "mano_colors": None,
            }

        # Create renderer
        self._renderer = OffscreenRenderer()
        for serial, K in zip(self._rs_serials, self._Ks):
            self._renderer.add_camera(K, serial)

        for object_id, mesh_file in zip(
            self._object_ids, self._reader.object_textured_files
        ):
            self._renderer.add_mesh(trimesh.load_mesh(str(mesh_file)), object_id)

    def render_poses(self, pose_type, pose_solver_folder=None):
        if pose_type not in [
            "joint_pose",
            "object_pose",
            "hand_pose",
            "fd_pose",
            "fd_pose_in_world",
            "fd_pose_in_cam",
        ]:
            self._logger.error(f"Invalid pose type: {pose_type}")
            return

        if pose_solver_folder is not None:
            pose_solver_folder = Path(pose_solver_folder)
        elif pose_type == "joint_pose":
            pose_solver_folder = self._data_folder / "processed/joint_pose_solver"
        elif pose_type == "object_pose":
            pose_solver_folder = self._data_folder / "processed/object_pose_solver"
        elif pose_type == "hand_pose":
            pose_solver_folder = self._data_folder / "processed/hand_pose_solver"
        elif pose_type in ["fd_pose", "fd_pose_in_world", "fd_pose_in_cam"]:
            pose_solver_folder = self._data_folder / "processed/fd_pose_solver"

        folder_name = pose_solver_folder.name

        if not pose_solver_folder.exists():
            self._logger.error(f"Pose solver folder not found: {pose_solver_folder}")
            return

        if pose_type == "joint_pose":
            self._load_joint_poses(pose_solver_folder)
        elif pose_type == "object_pose":
            self._load_object_poses(pose_solver_folder)
        elif pose_type == "hand_pose":
            self._load_hand_poses(pose_solver_folder)
        elif pose_type in ["fd_pose", "fd_pose_in_world"]:
            self._load_fd_poses_in_world(pose_solver_folder)
        elif pose_type in ["fd_pose", "fd_pose_in_cam"]:
            self._load_fd_poses_in_cam(pose_solver_folder)

        if pose_type in [
            "joint_pose",
            "object_pose",
            "hand_pose",
            "fd_pose",
            "fd_pose_in_world",
        ]:
            vis_images = self._render_poses_in_world(self._poses_dict)
            if vis_images is not None:
                self._logger.info("Saving vis images...")
                self._save_images(vis_images, pose_solver_folder / "vis" / folder_name)
                self._logger.info("Creating vis video...")
                create_video_from_rgb_images(
                    pose_solver_folder / f"vis_{folder_name}.mp4",
                    vis_images,
                    fps=30,
                )
                vis_images = None

        if pose_type in ["fd_pose", "fd_pose_in_cam"]:
            vis_images = self._render_fd_ob_in_cam_poses(self._poses_dict)
            if vis_images:
                self._logger.info("Saving vis images...")
                self._save_images(vis_images, pose_solver_folder / "vis" / "ob_in_cam")
                self._logger.info("Creating vis video...")
                create_video_from_rgb_images(
                    pose_solver_folder / f"vis_{folder_name}_ob_in_cam.mp4",
                    vis_images,
                    fps=30,
                )
                vis_images = None

        del vis_images
        gc.collect()

    def reset(self):
        self._poses_dict["ob_poses_in_world"] = None
        self._poses_dict["ob_poses_in_cam"] = None
        self._poses_dict["mano_verts"] = None
        self._poses_dict["mano_faces"] = None
        self._poses_dict["mano_colors"] = None

    def _load_joint_poses(self, pose_solver_folder):
        self._load_hand_poses(pose_solver_folder)
        self._load_object_poses(pose_solver_folder)

    def _load_hand_poses(self, pose_solver_folder):
        poses_m_file = pose_solver_folder / "poses_m.npy"
        if poses_m_file.exists():
            poses_m = np.load(poses_m_file)
            self._logger.debug(f"Loaded MANO poses from: {poses_m_file}")
            self._logger.debug(f"MANO poses: {poses_m.shape}")
        else:
            self._logger.error("MANO poses not found!!!")
            return

    def _load_object_poses(self, pose_solver_folder):
        self._poses_dict["ob_poses_in_world"] = None
        poses_o_file = pose_solver_folder / "poses_o.npy"
        poses_o_raw_file = pose_solver_folder / "poses_o_raw.npy"

        if poses_o_file.exists():
            ob_poses_in_world = np.load(poses_o_file)
            self._logger.debug(f"Loaded object poses from: {poses_o_file}")
        elif poses_o_raw_file.exists():
            ob_poses_in_world = np.load(poses_o_raw_file)
            self._logger.debug(f"Loaded object poses from: {poses_o_raw_file}")
        else:
            self._logger.error("Object poses not found!!!")
            return

        ob_poses_in_world = np.stack(
            [quat_to_mat(pose) for pose in ob_poses_in_world], axis=1
        )
        self._logger.debug(f"Object poses shape: {ob_poses_in_world.shape}")
        self._poses_dict["ob_poses_in_world"] = ob_poses_in_world

    def _load_fd_poses_in_world(self, pose_solver_folder):
        self._poses_dict["ob_poses_in_world"] = None
        poses_in_world_file = pose_solver_folder / "fd_poses.npy"
        poses_in_world_raw_file = pose_solver_folder / "fd_poses_raw.npy"
        if poses_in_world_file.exists():
            ob_poses_in_world = np.load(poses_in_world_file)
            self._logger.debug(f"Loaded ob_in_world poses from: {poses_in_world_file}")
        elif poses_in_world_raw_file.exists():
            ob_poses_in_world = np.load(poses_in_world_raw_file)
            self._logger.debug(
                f"Loaded ob_in_world poses from: {poses_in_world_raw_file}"
            )
        else:
            pose_in_world_files = [
                sorted(pose_solver_folder.glob(f"{object_id}/ob_in_world/*.txt"))
                for object_id in self._object_ids
            ]
            if not all(len(files) == self._num_frames for files in pose_in_world_files):
                self._logger.error(
                    "Number of ob_in_world pose files do not match the number of frames."
                )
                return
            else:
                ob_poses_in_world = [
                    [None] * self._num_frames for _ in self._object_ids
                ]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=CFG.max_workers
                ) as executor:
                    features = {
                        executor.submit(
                            read_pose_from_txt,
                            pose_in_world_files[o_idx][f_idx],
                        ): (o_idx, f_idx)
                        for f_idx in range(self._num_frames)
                        for o_idx in range(len(self._object_ids))
                    }
                    for future in concurrent.futures.as_completed(features):
                        try:
                            o_idx, f_idx = features[future]
                            ob_poses_in_world[o_idx][f_idx] = future.result()
                        except Exception as e:
                            self._logger.error(f"Error loading frame: {e}")
                    features.clear()  # clear the dictionary to free memory
                    del features
                self._logger.debug(f"Loaded ob_in_world poses from txt files")
        ob_poses_in_world = np.stack(
            [quat_to_mat(np.array(poses)) for poses in ob_poses_in_world], axis=1
        )
        self._poses_dict["ob_poses_in_world"] = ob_poses_in_world
        self._logger.debug(f"ob_poses_in_world: {ob_poses_in_world.shape}")

    def _load_fd_poses_in_cam(self, pose_solver_folder):
        self._poses_dict["ob_poses_in_cam"] = None
        poses_in_cam_file = pose_solver_folder / "fd_poses_in_cam.npy"
        if poses_in_cam_file.exists():
            ob_poses_in_cam = np.load(poses_in_cam_file)
            self._logger.debug(f"Loaded ob_in_cam poses from: {poses_in_cam_file}")
        else:
            pose_in_cam_files = [
                [
                    sorted(pose_solver_folder.glob(f"{o_id}/ob_in_cam/{s}/*.txt"))
                    for s in self._valid_serials
                ]
                for o_id in self._object_ids
            ]
            if not all(
                len(pose_in_cam_files[o_idx][s_idx]) == num_frames
                for s_idx in range(len(valid_serials))
                for o_idx in range(len(object_ids))
            ):
                logger.error(
                    "Number of ob_in_cam pose files do not match the number of frames."
                )
                return
            else:
                ob_poses_in_cam = [
                    [[None] * self._num_frames for _ in self._valid_serials]
                    for _ in self._object_ids
                ]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=CFG.max_workers
                ) as executor:
                    features = {
                        executor.submit(
                            read_pose_from_txt,
                            pose_in_cam_files[o_idx][s_idx][f_idx],
                        ): (o_idx, s_idx, f_idx)
                        for f_idx in range(self._num_frames)
                        for s_idx in range(len(self._valid_serials))
                        for o_idx in range(len(self._object_ids))
                    }
                    for future in concurrent.futures.as_completed(features):
                        try:
                            o_idx, s_idx, f_idx = features[future]
                            ob_poses_in_cam[o_idx][s_idx][f_idx] = future.result()
                        except Exception as e:
                            self._logger.error(f"Error loading frame: {e}")
                    features.clear()  # clear the dictionary to free memory
                    del features
                self._logger.debug(f"Loaded ob_in_cam poses from txt files")
        ob_poses_in_cam = np.stack(
            [
                [quat_to_mat(poses) for poses in serial_poses]
                for serial_poses in ob_poses_in_cam
            ],
            axis=0,
        )
        self._poses_dict["ob_poses_in_cam"] = ob_poses_in_cam
        self._logger.debug(f"ob_poses_in_cam: {ob_poses_in_cam.shape}")

    def _render_fd_ob_in_cam_pose_frame(self, frame_id, ob_in_cam_poses):
        color_images = [
            self._reader.get_color(serial, frame_id) for serial in self._valid_serials
        ]
        render_images = [
            self._renderer.get_render_colors(
                width=self._rs_width,
                height=self._rs_height,
                cam_names=serial,
                cam_poses=np.eye(4),
                mesh_names=self._object_ids,
                mesh_poses=ob_in_cam_poses[:][serial_idx],
            )
            for serial_idx, serial in enumerate(self._valid_serials)
        ]
        vis_image = draw_image_grid(
            images=[
                draw_image_overlay(color_image, render_image, 0.7)
                for color_image, render_image in zip(color_images, render_images)
            ],
            names=self._valid_serials,
            max_cols=len(self._valid_serials) // 2,
            facecolor="black",
            titlecolor="white",
        )
        return vis_image

    def _render_fd_ob_in_cam_poses(self, poses_dict):
        self._logger.info(f"Rendering fd ob_in_cam vis images...")
        if poses_dict["ob_poses_in_cam"] is None:
            self._logger.error("No ob_in_cam poses found!!!")
            return None

        vis_images = [None] * self._num_frames
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CFG.max_workers
        ) as executor:
            features = {
                executor.submit(
                    self._render_fd_ob_in_cam_pose_frame,
                    frame_id,
                    poses_dict["ob_poses_in_cam"][:][:][frame_id],
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(features):
                try:
                    frame_id = features[future]
                    vis_images[frame_id] = future.result()
                    tqbar.update(1)
                except Exception as e:
                    self._logger.error(f"Error rendering frame: {e}")
        tqbar.close()
        return vis_images

    def _render_world_frame(
        self, frame_id, ob_poses_in_world=None, mano_verts_in_world=None
    ):
        color_images = [
            self._reader.get_color(serial, frame_id) for serial in self._rs_serials
        ]
        render_images = self._renderer.get_render_colors(
            width=self._rs_width,
            height=self._rs_height,
            cam_names=self._rs_serials,
            cam_poses=self._camera_poses,
            mesh_names=self._object_ids,
            mesh_poses=ob_poses_in_world,
            mano_vertices=mano_verts_in_world,
            mano_faces=self._poses_dict["mano_faces"],
            mano_colors=self._poses_dict["mano_colors"],
        )
        vis_image = draw_image_grid(
            images=[
                draw_image_overlay(color_image, render_image, 0.7)
                for color_image, render_image in zip(color_images, render_images)
            ],
            names=self._rs_serials,
            max_cols=4,
            facecolor="black",
            titlecolor="white",
        )
        return vis_image

    def _render_poses_in_world(self, poses_dict):
        self._logger.info("Rendering vis images...")

        if poses_dict["ob_poses_in_world"] is None and poses_dict["mano_verts"] is None:
            self._logger.error("Neither ob_in_world nor mano poses found!!!")
            return None

        ob_poses_in_world = poses_dict["ob_poses_in_world"]
        mano_verts = poses_dict["mano_verts"]

        vis_images = [None] * self._num_frames
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CFG.max_workers
        ) as executor:
            features = {
                executor.submit(
                    self._render_world_frame,
                    frame_id,
                    (
                        ob_poses_in_world[frame_id]
                        if ob_poses_in_world is not None
                        else None
                    ),
                    mano_verts[frame_id] if mano_verts is not None else None,
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(features):
                try:
                    frame_id = features[future]
                    vis_images[frame_id] = future.result()
                    tqbar.update(1)
                except Exception as e:
                    self._logger.error(f"Error rendering frame: {e}")
        tqbar.close()
        return vis_images

    def _save_images(self, vis_images, save_folder):
        make_clean_folder(save_folder)
        tqbar = tqdm(total=self._num_frames, ncols=80)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=CFG.max_workers
        ) as executor:
            features = {
                executor.submit(
                    write_rgb_image,
                    save_folder / f"vis_{frame_id:06d}.png",
                    vis_images[frame_id],
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(features):
                try:
                    frame_id = features[future]
                    future.result()
                    tqbar.update(1)
                except Exception as e:
                    self._logger.error(f"Error saving frame: {e}")
        tqbar.close()

    def render_hand_poses(
        self, verts_m, faces_m, colors_m, save_vis_folder, save_video_file
    ):
        self._logger.info("Rendering hand poses...")
        self.reset()
        self._poses_dict["mano_verts"] = verts_m
        self._poses_dict["mano_faces"] = faces_m
        self._poses_dict["mano_colors"] = colors_m

        vis_images = self._render_poses_in_world(self._poses_dict)
        if vis_images is not None:
            self._logger.info("Saving vis images...")
            self._save_images(vis_images, save_vis_folder)
            self._logger.info("Creating vis video...")
            create_video_from_rgb_images(save_video_file, vis_images, fps=30)
            vis_images = None
        del vis_images
        gc.collect()

    def render_object_poses(self, ob_poses, save_vis_folder, save_video_file):
        self._logger.info("Rendering object poses...")
        self.reset()
        self._poses_dict["ob_poses_in_world"] = ob_poses

        vis_images = self._render_poses_in_world(self._poses_dict)
        if vis_images is not None:
            self._logger.info("Saving vis images...")
            self._save_images(vis_images, save_vis_folder)
            self._logger.info("Creating vis video...")
            create_video_from_rgb_images(save_video_file, vis_images, fps=30)
            vis_images = None
        del vis_images
        gc.collect()

    def render_joint_poses(
        self, ob_poses, verts_m, faces_m, colors_m, save_vis_folder, save_video_file
    ):
        self._logger.info("Rendering joint poses...")
        self.reset()
        self._poses_dict["ob_poses_in_world"] = ob_poses
        self._poses_dict["mano_verts"] = verts_m
        self._poses_dict["mano_faces"] = faces_m
        self._poses_dict["mano_colors"] = colors_m

        vis_images = self._render_poses_in_world(self._poses_dict)
        if vis_images is not None:
            self._logger.info("Saving vis images...")
            self._save_images(vis_images, save_vis_folder)
            self._logger.info("Creating vis video...")
            create_video_from_rgb_images(save_video_file, vis_images, fps=30)
            vis_images = None
        del vis_images
        gc.collect()
