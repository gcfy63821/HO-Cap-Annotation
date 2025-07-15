import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from pprint import pformat
from hocap_annotation.utils import *
from hocap_annotation.loss import (
    MeshSDFLoss,
    Keypoint2DLoss,
    Keypoint3DLoss,
    MANORegLoss,
    PoseSmoothnessLoss,
)
from hocap_annotation.loaders import SequenceLoader
from hocap_annotation.rendering import HOCapRenderer


class ManoPoseSolver:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._data_folder = Path(sequence_folder)
        self._debug = debug
        self._device = CFG.device
        self._save_folder = self._data_folder / "processed/hand_pose_solver"
        self._save_folder.mkdir(parents=True, exist_ok=True)

        self._log_file = self._save_folder / "mano_pose_solver.log"
        # Remove the existing log file
        if self._log_file.exists():
            self._log_file.unlink()
        self._logger = get_logger(
            self.__class__.__name__, "DEBUG" if debug else "INFO", self._log_file
        )

        self._log_info_steps = 500
        self._log_debug_steps = 100

        # Load parameters from data loader
        self._check_required_pose_files()

        # Load optimization configuration
        self._load_optim_config()

        # Load parameters from data loader
        self._load_dataloader_params()

    def _load_optim_config(self):
        self._logger.info("Loading optimization configuration...")
        optim_config = CFG.optimization.hand_pose_solver
        self._lr = optim_config["lr"]
        self._total_steps = optim_config["total_steps"]
        self._sdf_steps = optim_config["sdf_steps"]
        self._smooth_steps = optim_config["smooth_steps"]
        self._w_sdf = optim_config["w_sdf"]
        self._w_kpt_2d = optim_config["w_kpt_2d"]
        self._w_kpt_3d = optim_config["w_kpt_3d"]
        self._w_reg = optim_config["w_reg"]
        self._w_smooth = optim_config["w_smooth"]
        self._w_smooth_rot = optim_config["w_smooth_rot"]
        self._w_smooth_trans = optim_config["w_smooth_trans"]
        self._w_smooth_acc_rot = optim_config["w_smooth_acc_rot"]
        self._w_smooth_acc_trans = optim_config["w_smooth_acc_trans"]
        self._win_size = optim_config["smooth_window_size"]
        self._dist_thresh = optim_config["sdf_dist_thresh"]
        self._load_offline_dpts = optim_config["load_offline_dpts"]
        self._valid_rs_serials = optim_config["valid_rs_serials"]
        self._valid_mano_joint_indices = optim_config["valid_mano_joint_indices"]
        self._logger.debug(
            "Optimization Config:\n" + pformat(optim_config, sort_dicts=False)
        )

    def _check_required_pose_files(self):
        self._logger.info("Checking existence of required files...")
        self._joints_2d_file = (
            self._data_folder / "processed/hand_detection/mp_handmarks_results.npz"
        )
        self._joints_3d_file = (
            self._data_folder / "processed/hand_detection/mp_joints_3d_raw.npy"
        )
        if not self._joints_2d_file.exists():
            msg = "Hand 2D Joints Detection Results do not exist!"
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            self._logger.info(f"Hand 2D keypoints file: {self._joints_2d_file}")
        if not self._joints_3d_file.exists():
            msg = "Hand 3D Joints Detection Results do not exist!"
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            self._logger.info(f"Hand 3D joints file: {self._joints_3d_file}")
        return

    def _load_dataloader_params(self):
        self._data_loader = SequenceLoader(
            self._data_folder, load_mano=True, device=self._device
        )
        self._num_frames = self._data_loader.num_frames
        self._rs_serials = self._data_loader.rs_serials
        self._mano_sides = self._data_loader.mano_sides
        self._rs_img_size = torch.tensor(
            [self._data_loader.rs_width - 1, self._data_loader.rs_height - 1],
            device=self._device,
            dtype=torch.float32,
        )
        M = self._data_loader.M2world
        self._M = M[[self._rs_serials.index(s) for s in self._valid_rs_serials]]
        self._valid_kpt_ind_m = [
            i * 21 + j
            for i in range(len(self._mano_sides))
            for j in self._valid_mano_joint_indices
        ]
        self._mano_group_layer = self._data_loader.mano_group_layer

    def _load_hand_2d_keypoints(self, kpts_file):
        kpts = np.load(kpts_file)
        kpts = np.stack([kpts[s] for s in self._valid_rs_serials], axis=1)
        kpts = kpts.transpose(2, 1, 0, 3, 4)  # (N, C, H, 2)
        kpts = np.concatenate(
            [kpts[:, :, 0 if side == "right" else 1] for side in self._mano_sides],
            axis=2,
        )  # (N, C, 21 * H, 2)

        invalid_kpt_indices = [
            i * 21 + j
            for i in range(len(self._mano_sides))
            for j in range(21)
            if j not in self._valid_mano_joint_indices
        ]
        valid_mask = np.all(kpts != -1, axis=-1)  # (N, C, 21 * H)
        valid_mask[:, :, invalid_kpt_indices] = False
        kpts = torch.from_numpy(kpts).to(self._device) / self._rs_img_size
        valid_mask = torch.from_numpy(valid_mask).to(self._device)
        self._logger.info(
            f"Hand 2D keypoints loaded: {kpts.shape}, valid_mask: {valid_mask.shape}"
        )
        return kpts, valid_mask

    def _load_hand_3d_joints(self, file_path):
        joints = np.load(file_path).astype(np.float32)  # (H, N, 21, 3)
        joints[:, 1] = joints[:, 0]
        joints = np.stack(
            [joints[0 if side == "right" else 1] for side in self._mano_sides], axis=0
        )  # (num_hands, num_frames, 21, 3)
        self._logger.info(f"Hand 3D joints loaded: {joints.shape}")
        return joints

    def _load_poses_m(self, pose_file):
        poses = np.load(pose_file).astype(np.float32)
        poses = np.stack(
            [poses[0 if side == "right" else 1] for side in self._mano_sides], axis=0
        )  # (num_hands, num_frames, 51)
        self._logger.info(f"MANO poses loaded: {poses.shape}")
        return poses

    def _mano_group_layer_forward(self, poses_m, subset=None):
        p = torch.cat(poses_m, dim=1)
        v, j = self._mano_group_layer(p, subset)
        if v.size(0) == 1:
            v = v.squeeze(0)
            j = j.squeeze(0)
        return v, j

    def _project_3d_to_2d(self, p_3d, Ms):
        """
        Project a batch of 3D points to 2D across multiple camera views in parallel.

        Parameters:
        - p_3d: A torch tensor of 3D points of shape (N, P, 3).
        - Ms: Camera projection matrices, torch tensor of shape (C, 3, 4).

        Returns:
        - projected_2d: Projected 2D points, torch tensor of shape (N, C, P, 2).
        """
        N, P = p_3d.shape[:2]
        C = Ms.shape[0]
        ones = torch.ones((N, P, 1), device=self._device, dtype=torch.float32)
        p_3d_hom = torch.cat((p_3d, ones), dim=2)  # Shape: (N, P, 4)
        # Prepare p_3d for broadcasting by adding a camera dimension
        p_3d_hom = p_3d_hom.unsqueeze(1)  # Shape: (N, 1, P, 4)
        # Broadcast p_3d to all cameras
        p_3d_hom = p_3d_hom.expand(-1, C, -1, -1)  # Shape: (N, C, P, 4)
        # Project points using camera matrices
        p_2d_hom = torch.einsum("cij, nckj->ncki", Ms, p_3d_hom)  # Shape: (N, C, P, 3)
        # Normalize the points to convert to 2D coordinates
        p_2d = p_2d_hom[..., :2] / p_2d_hom[..., 2:3]  # Shape: (N, C, P, 2)
        return p_2d

    def _get_dpts_for_loss_sdf(self, verts, faces, dpts, dist_thresh):
        _, dist, _ = self._meshsdf_loss(verts, faces, dpts)
        return dpts[dist < dist_thresh]

    def _loss_sdf(self, verts_list, faces, dpts_list):
        def loss_sdf(verts, faces, dpts):
            if dpts.size(0) < 2000:
                return self._zero
            loss, _, _ = self._meshsdf_loss(verts, faces, dpts)
            loss *= 1e3  # scale to meters
            return loss

        if len(verts_list) != len(dpts_list):
            msg = f"Length mismatch: verts_list has {len(verts_list)} items, dpts_list has {len(dpts_list)}."
            self._logger.error(msg)
            raise ValueError(msg)

        losses = [None] * len(verts_list)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(loss_sdf, verts, faces, dpts): i
                for i, (verts, dpts) in enumerate(zip(verts_list, dpts_list))
            }
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                losses[i] = future.result()
            futures.clear()

        total_loss = torch.stack(losses, dim=0)
        total_loss = total_loss.sum() / len(verts_list)
        return total_loss

    def _initialize_pose_m_from_3d_joints(self, joints_3d):
        pose_m = np.zeros(
            (self._mano_group_layer.num_obj, self._num_frames, 51), dtype=np.float32
        )
        # Update the global translations
        pose_m[:, :, -3:] = joints_3d[:, :, 0].copy()
        pose_m = [
            torch.nn.Parameter(torch.from_numpy(p).to(self._device), requires_grad=True)
            for p in pose_m
        ]
        return pose_m

    def _prepare_dpts_list_for_loss_sdf(self, subset):
        self._logger.info(f"Preparing dpts for SDF loss...")
        save_dpts_folder = self._save_folder / "dpts"
        save_dpts_folder.mkdir(parents=True, exist_ok=True)

        dpts_files = sorted(save_dpts_folder.glob("dpts_*.ply"))
        dpts_list = [None] * self._num_frames
        if self._load_offline_dpts and len(dpts_files) == self._num_frames:
            self._logger.info(f"Loading offline dpts...")
            tqbar = tqdm(total=self._num_frames, ncols=100)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(read_points_from_ply, dpts_f): idx
                    for idx, dpts_f in enumerate(dpts_files)
                }
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    dpts_list[idx] = torch.from_numpy(future.result()).to(self._device)
                    tqbar.update(1)
                futures.clear()
            tqbar.close()
        else:
            self._logger.info(f"Generating dpts...")
            faces, _ = self._mano_group_layer.get_f_from_inds(subset)
            verts, _ = self._mano_group_layer_forward(self._pose_m, subset)
            for f_idx in tqdm(range(self._num_frames), ncols=100):
                self._data_loader.step_by_frame_id(f_idx)
                points = self._data_loader.points[self._data_loader.masks]
                print(f"Masks size for frame {f_idx}: {self._data_loader.masks.size()}")
                print(f"Points  _data_loader.points for frame {f_idx}: {points.size()}")
                points = self._get_dpts_for_loss_sdf(
                    verts[f_idx], faces, points, self._dist_thresh
                )
                print(f"Points after _get_dpts_for_loss_sdf for frame {f_idx}: {points.size()}")
                points = process_points(
                    points=points, voxel_size=0.003, remove_outliers=True
                )
                if points.size(0) == 0:
                    self._logger.warning(
                        f"No valid dpts for frame {f_idx}, using zeros."
                    )
                    points = torch.zeros(
                        (0, 3), dtype=torch.float32, device=self._device
                    )
                dpts_list[f_idx] = points

            # Save dpts to files
            self._logger.info("Saving dpts to files...")
            tqbar = tqdm(total=self._num_frames, ncols=100)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        write_points_to_ply,
                        dpts_list[f_idx].cpu().numpy(),
                        save_dpts_folder / f"dpts_{f_idx:06d}.ply",
                    ): f_idx
                    for f_idx in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    tqbar.update(1)
                futures.clear()
            tqbar.close()
        self._logger.info("Done preparing dpts for SDF loss.")
        return dpts_list

    def _save_log_loss(self, save_name):
        self._logger.info("Saving loss log...")
        np.savez(
            self._save_folder / f"{save_name}.npz",
            total=self._log_loss[0],
            sdf=self._log_loss[1],
            kpt_2d=self._log_loss[2],
            kpt_3d=self._log_loss[3],
            reg=self._log_loss[4],
            smooth=self._log_loss[5],
        )
        loss_curve_img = draw_losses_curve(
            self._log_loss, ["total", "sdf", "kpt_2d", "kpt_3d", "reg", "smooth"]
        )
        write_rgb_image(self._save_folder / f"{save_name}_curve.png", loss_curve_img)

    def _save_optimized_poses_m(self, save_name="poses_m"):
        self._logger.info("Saving optimized hand poses...")
        optim_pose_m = torch.stack(
            [p.detach().cpu() for p in self._pose_m], dim=0
        ).clone()
        optim_pose_m = optim_pose_m.numpy().astype(np.float32)
        if len(self._mano_sides) == 1:
            filler_pose_m = np.full_like(optim_pose_m, -1)
            if "right" in self._mano_sides:
                optim_pose_m = np.concatenate([optim_pose_m, filler_pose_m], axis=0)
            elif "left" in self._mano_sides:
                optim_pose_m = np.concatenate([filler_pose_m, optim_pose_m], axis=0)
        self._logger.debug(f"optim_pose_m: {optim_pose_m.shape}")
        np.save(self._save_folder / f"{save_name}.npy", optim_pose_m)

    def initialize_optimizer(self):
        self._logger.info("Initializing optimizer...")
        self._meshsdf_loss = MeshSDFLoss(loss_type="l2").to(self._device)
        self._loss_kpt_2d_m = Keypoint2DLoss(loss_type="l2_norm").to(self._device)
        self._loss_kpt_3d_m = Keypoint3DLoss(loss_type="l2_norm").to(self._device)
        self._loss_reg_m = MANORegLoss().to(self._device)
        self._loss_smooth = PoseSmoothnessLoss(
            win_size=self._win_size,
            w_vel_r=self._w_smooth_rot,
            w_vel_t=self._w_smooth_trans,
            w_acc_r=self._w_smooth_acc_rot,
            w_acc_t=self._w_smooth_acc_trans,
        ).to(self._device)
        self._zero = torch.zeros((), dtype=torch.float32, device=self._device)

        joints_3d = self._load_hand_3d_joints(self._joints_3d_file)

        self._pose_m = self._initialize_pose_m_from_3d_joints(joints_3d)
        self._optimizer = torch.optim.Adam(self._pose_m, lr=self._lr)

        if self._w_kpt_3d > 0:
            joints_3d = np.concatenate([p for p in joints_3d], axis=1)
            self._target_joints_3d = torch.from_numpy(joints_3d).to(self._device)
            self._logger.info(f"target_joints_3d: {self._target_joints_3d.shape}")

        # loss, sdf, kpt_2d, kpt_3d, reg, smooth
        self._log_loss = np.zeros((6, self._total_steps), dtype=np.float32)

        # Load 2d keypoints for hands
        if self._w_kpt_2d > 0:
            self._target_kpts_m, self._valid_mask_m = self._load_hand_2d_keypoints(
                self._joints_2d_file
            )

    def solve(self):
        subset_m = list(range(self._mano_group_layer.num_obj))

        self._logger.info(">>>>>>>>>> Start optimization <<<<<<<<<<")
        t_s = time.time()
        faces_m, _ = self._mano_group_layer.get_f_from_inds(subset_m)

        tt_s = time.time()
        ttt_s = time.time()
        for step in range(self._total_steps):
            # Reset gradients
            self._optimizer.zero_grad()

            # Get MANO vertices and joints
            verts_m, joints_m = self._mano_group_layer_forward(self._pose_m, subset_m)

            # Prepare dpts for SDF loss
            if self._w_sdf > 0 and step == self._total_steps - self._sdf_steps:
                self.save_results(loss_name="loss_kpt", poses_m_name="poses_m_kpt")
                # self.render_optimized_poses(
                #     video_name="vis_hand_pose_solver_kpt", poses_m_name="poses_m_kpt"
                # )
                dpts_list = self._prepare_dpts_list_for_loss_sdf(subset_m)
                tt_s = time.time()
                ttt_s = time.time()
                self._log_info_steps = 10
                self._log_debug_steps = 1

            # Calculate losses
            if self._w_kpt_2d == 0:
                loss_kpt_2d = self._zero
            else:
                pred_kpts_m = (
                    self._project_3d_to_2d(joints_m, self._M) / self._rs_img_size
                )
                loss_kpt_2d = self._loss_kpt_2d_m(
                    pred_kpts_m, self._target_kpts_m, self._valid_mask_m
                )
                loss_kpt_2d *= self._w_kpt_2d

            if self._w_kpt_3d == 0:
                loss_kpt_3d = self._zero
            else:
                loss_kpt_3d = self._loss_kpt_3d_m(
                    joints_m, self._target_joints_3d, self._valid_kpt_ind_m
                )
                loss_kpt_3d *= self._w_kpt_3d

            if self._w_sdf == 0:
                loss_sdf = self._zero
            elif step >= self._total_steps - self._sdf_steps:
                loss_sdf = self._loss_sdf(verts_m, faces_m, dpts_list)
                loss_sdf *= self._w_sdf
            else:
                loss_sdf = self._zero

            if self._w_reg == 0:
                loss_reg = self._zero
            else:
                loss_reg = self._loss_reg_m(self._pose_m, subset=subset_m)
                loss_reg *= self._w_reg

            if self._w_smooth == 0:
                loss_smooth = self._zero
            elif step >= self._total_steps - self._smooth_steps:
                loss_smooth = self._loss_smooth(self._pose_m, subset_m)
                loss_smooth *= self._w_smooth
            else:
                loss_smooth = self._zero

            loss = loss_sdf + loss_kpt_2d + loss_kpt_3d + loss_reg + loss_smooth

            # Set grad to None to prevent Adam from updating the parameters even when
            # the grad are all zeros. An alternative is to place this before the
            # forward pass, but the currently implementation of group layers does not
            # allow this since the non-active poses will still always be used due to
            # concatenation and hence their grad will be 0.0 rather than None.
            for i, p in enumerate(self._pose_m):
                if i not in subset_m and p.grad is not None:
                    assert torch.all(p.grad == 0.0)
                    p.grad = None

            loss.backward()
            self._optimizer.step()

            self._log_loss[:, step] = [
                loss.item(),
                loss_sdf.item(),
                loss_kpt_2d.item(),
                loss_kpt_3d.item(),
                loss_reg.item(),
                loss_smooth.item(),
            ]

            log_msg = (
                f"step: {step+1:04d}/{self._total_steps:04d} "
                + f"| loss: {loss.item():11.8f} "
                + f"| sdf: {loss_sdf.item():11.8f} "
                + f"| kpt_2d: {loss_kpt_2d.item():11.8f} "
                + f"| kpt_3d: {loss_kpt_3d.item():11.8f} "
                + f"| reg: {loss_reg.item():11.8f} "
                + f"| smooth: {loss_smooth.item():11.8f} "
            )
            if (step + 1) % self._log_info_steps == 0:
                self._logger.info(log_msg + f"| time: {time.time() - tt_s:.2f}s")
                tt_s = time.time()
            elif (step + 1) % self._log_debug_steps == 0:
                self._logger.debug(log_msg + f"| time: {time.time() - ttt_s:.2f}s")
                ttt_s = time.time()

        self._logger.info(
            f">>>>>>>>>> Optimization Done! ({time.time() - t_s:.2f}s) <<<<<<<<<<"
        )

    def save_results(self, loss_name="loss", poses_m_name="poses_m"):
        self._logger.info(">>>>>>>>>> Saving results <<<<<<<<<<")
        # Save loss log
        self._save_log_loss(loss_name)

        # Save optimized poses
        self._save_optimized_poses_m(poses_m_name)

        self._logger.info(">>>>>>>>>> Saving results Done!!! <<<<<<<<<<")

    def render_optimized_poses(
        self, video_name="vis_hand_pose_solver", poses_m_name="poses_m"
    ):
        self._logger.debug(">>>>>>>>>> Rendering optimized poses <<<<<<<<<<")
        t_s = time.time()
        poses_o, verts_m, faces_m, colors_m, joints_m = None, None, None, None, None

        # Prepare hand data
        poses_m = self._load_poses_m(self._save_folder / f"{poses_m_name}.npy")
        poses_m = [torch.from_numpy(p).to(self._device) for p in poses_m]
        verts_m, joints_m = self._mano_group_layer_forward(poses_m)
        verts_m = verts_m.detach().clone().cpu().numpy()
        joints_m = joints_m.detach().clone().cpu().numpy()
        faces_m = [self._mano_group_layer.f.detach().clone().cpu().numpy()]
        colors_m = []
        for i, side in enumerate(self._mano_sides):
            faces_m.append(np.array(NEW_MANO_FACES[side]) + i * NUM_MANO_VERTS)
            colors_m.append(HAND_COLORS[1 if side == "right" else 2].rgb_norm)
        faces_m = np.concatenate(faces_m, axis=0).astype(np.int64)

        # Render poses
        renderer = HOCapRenderer(self._data_folder, log_file=self._log_file)
        renderer.update_render_dict(poses_o, verts_m, faces_m, colors_m, joints_m)
        renderer.render_pose_images(
            save_folder=self._save_folder / f"vis",
            save_video_path=self._save_folder / f"{video_name}.mp4",
            vis_only=True,
            save_vis=True,
        )
        self._logger.debug(
            f">>>>>>>>>> Rendering Done!!! ({time.time() - t_s:.2f}s) <<<<<<<<<<"
        )

    def run(self):
        self._logger.info("=" * 100)
        self._logger.info("Start Hand Pose Solver")
        self._logger.info("=" * 100)
        t_s = time.time()
        # Initialize optimizer
        self.initialize_optimizer()

        # Start optimization
        self.solve()

        # Save results
        self.save_results()

        # Render optimized poses
        # self.render_optimized_poses()

        self._logger.info("=" * 100)
        self._logger.info(f"Hand Pose Solver Done! ({time.time() - t_s:.2f}s)")
        self._logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Pose Solver")
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Whether to enable debug mode"
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder!")

    solver = ManoPoseSolver(args.sequence_folder, args.debug)
    solver.run()
