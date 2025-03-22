import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from pprint import pformat
from hocap_annotation.utils import *
from hocap_annotation.loss import (
    MeshSDFLoss,
    PoseAlignmentLoss,
    PoseSmoothnessLoss,
)
from hocap_annotation.loaders import SequenceLoader
from hocap_annotation.rendering import HOCapRenderer


class JointPoseSolver:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._data_folder = Path(sequence_folder)
        self._debug = debug
        self._device = CFG.device
        self._save_folder = self._data_folder / "processed" / "joint_pose_solver"
        self._save_folder.mkdir(parents=True, exist_ok=True)

        self._log_file = self._save_folder / "joint_pose_solver.log"
        # Remove the existing log file
        if self._log_file.exists():
            self._log_file.unlink()
        self._logger = get_logger(
            self.__class__.__name__, "DEBUG" if debug else "INFO", self._log_file
        )

        self._log_info_steps = 10
        self._log_debug_steps = 1

        # Load optimization config
        self._load_optim_config()

        # Check if the required files exist
        self._check_required_files()

        # Load parameters from data loader
        self._load_dataloader_params()

    def _load_optim_config(self):
        self._logger.info("Loading optimization configuration...")
        optim_config = CFG.optimization.joint_pose_solver
        self._lr = optim_config["lr"]
        self._total_steps = optim_config["total_steps"]
        self._sdf_steps = optim_config["sdf_steps"]
        self._w_sdf = optim_config["w_sdf"]
        self._w_reg_m = optim_config["w_reg_m"]
        self._w_reg_o = optim_config["w_reg_o"]
        self._w_smooth_m = optim_config["w_smooth_m"]
        self._w_smooth_rot_m = optim_config["w_smooth_rot_m"]
        self._w_smooth_trans_m = optim_config["w_smooth_trans_m"]
        self._w_smooth_acc_rot_m = optim_config["w_smooth_acc_rot_m"]
        self._w_smooth_acc_trans_m = optim_config["w_smooth_acc_trans_m"]
        self._win_size_m = optim_config["smooth_window_size_m"]
        self._w_smooth_o = optim_config["w_smooth_o"]
        self._w_smooth_rot_o = optim_config["w_smooth_rot_o"]
        self._w_smooth_trans_o = optim_config["w_smooth_trans_o"]
        self._w_smooth_acc_rot_o = optim_config["w_smooth_acc_rot_o"]
        self._w_smooth_acc_trans_o = optim_config["w_smooth_acc_trans_o"]
        self._win_size_o = optim_config["smooth_window_size_o"]
        self._dist_thresh = optim_config["sdf_dist_thresh"]
        self._load_offline_dpts = optim_config["load_offline_dpts"]
        self._logger.debug(
            "Optimization Config:\n" + pformat(optim_config, sort_dicts=False)
        )

    def _check_required_files(self):
        self._logger.info("Checking existence of required files...")
        self._pose_o_file = (
            self._data_folder / "processed" / "object_pose_solver" / "poses_o.npy"
        )
        self._pose_m_file = (
            self._data_folder / "processed" / "hand_pose_solver" / "poses_m.npy"
        )
        msg = "File not found: {}"
        if not self._pose_o_file.exists():
            self._logger.error(msg.format(self._pose_o_file))
            raise FileNotFoundError(msg.format(self._pose_o_file))
        else:
            self._logger.info(f"Object poses file: {self._pose_o_file}")

        if not self._pose_m_file.exists():
            self._logger.error(msg.format(self._pose_m_file))
            raise FileNotFoundError(msg.format(self._pose_m_file))
        else:
            self._logger.info(f"Hand poses file: {self._pose_m_file}")
        return

    def _load_dataloader_params(self):
        self._data_loader = SequenceLoader(
            self._data_folder, load_mano=True, load_object=True, device=self._device
        )
        self._num_frames = self._data_loader.num_frames
        self._rs_serials = self._data_loader.rs_serials
        self._mano_sides = self._data_loader.mano_sides
        self._mano_group_layer = self._data_loader.mano_group_layer
        self._object_group_layer = self._data_loader.object_group_layer

    def _load_poses_o(self, pose_file):
        poses = np.load(pose_file).astype(np.float32)
        self._logger.debug(f"Object poses loaded: {poses.shape}")
        return poses

    def _load_poses_m(self, pose_file):
        poses = np.load(pose_file).astype(np.float32)
        poses = np.stack(
            [poses[0 if side == "right" else 1] for side in self._mano_sides], axis=0
        )  # (num_hands, num_frames 51)
        self._logger.info(f"MANO poses loaded: {poses.shape}")
        return poses

    def _object_group_layer_forward(self, pose_o, subset=None):
        p = torch.cat(pose_o, dim=1)
        v, n = self._object_group_layer(p, subset)
        if v.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

    def _mano_group_layer_forward(self, poses_m, subset=None):
        p = torch.cat(poses_m, dim=1)
        v, j = self._mano_group_layer(p, subset)
        if v.size(0) == 1:
            v = v.squeeze(0)
            j = j.squeeze(0)
        return v, j

    def _get_dpts_for_loss_sdf(self, verts, faces, dpts, dist_thresh):
        _, dist, _ = self._meshsdf_loss(verts, faces, dpts)
        return dpts[dist < dist_thresh]

    def _loss_sdf(self, verts_list, faces, dpts_list):
        def loss_sdf(verts, faces, dpts):
            if dpts.size(0) < 500:
                return self._zero
            loss, _, _ = self._meshsdf_loss(verts, faces, dpts)
            loss *= 1e3  # Scale to meters
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
                try:
                    losses[i] = future.result()
                except Exception as e:
                    self._logger.error(f"Error in loss_sdf: {e}")
                    losses[i] = self._zero
        total_loss = torch.stack(losses, dim=0)
        total_loss = total_loss.sum() / len(verts_list)
        return total_loss

    def _save_log_loss(self, save_name="loss"):
        self._logger.info("Saving loss log...")
        np.savez(
            self._save_folder / f"{save_name}.npz",
            total=self._log_loss[0],
            sdf=self._log_loss[1],
            reg_m=self._log_loss[2],
            smooth_m=self._log_loss[3],
            reg_o=self._log_loss[4],
            smooth_o=self._log_loss[5],
        )
        loss_curve_img = draw_losses_curve(
            self._log_loss, ["total", "sdf", "reg_m", "smooth_m", "reg_o", "smooth_o"]
        )
        write_rgb_image(self._save_folder / f"{save_name}_curve.png", loss_curve_img)

    def _save_optimized_poses_m(self, save_name="poses_m"):
        self._logger.info("Saving optimized hand poses...")
        optim_pose_m = torch.stack([p.data for p in self._pose_m], dim=1).squeeze(0)
        optim_pose_m = optim_pose_m.cpu().numpy().astype(np.float32)
        optim_pose_m = optim_pose_m.swapaxes(0, 1)  # (num_hands, num_frames, 51)
        if len(self._mano_sides) == 1:
            filler_pose_m = np.full_like(optim_pose_m, -1)
            if "right" in self._mano_sides:
                optim_pose_m = np.concatenate([optim_pose_m, filler_pose_m], axis=0)
            elif "left" in self._mano_sides:
                optim_pose_m = np.concatenate([filler_pose_m, optim_pose_m], axis=0)
        self._logger.debug(f"optim_pose_m: {optim_pose_m.shape}")
        np.save(self._save_folder / f"{save_name}.npy", optim_pose_m)

    def _save_optimized_poses_o(self, save_name="poses_o"):
        self._logger.info("Saving optimized object poses...")
        optim_pose_o = torch.stack([p.data for p in self._pose_o], dim=0).squeeze(0)
        optim_pose_o = optim_pose_o.cpu().numpy().astype(np.float32)
        optim_pose_o = np.stack([rvt_to_quat(ps) for ps in optim_pose_o])
        self._logger.debug(f"optim_pose_o: {optim_pose_o.shape}")
        np.save(self._save_folder / f"{save_name}.npy", optim_pose_o)

    def _initialize_pose_m_from_poses(self, poses_m):
        pose_m = [
            torch.nn.Parameter(
                torch.from_numpy(poses_m[i]).to(self._device), requires_grad=True
            )
            for i in range(self._mano_group_layer.num_obj)
        ]
        return pose_m

    def _initialize_pose_o_from_poses(self, poses_o):
        pose_o = [
            torch.nn.Parameter(
                torch.from_numpy(quat_to_rvt(poses_o[i])).to(self._device),
                requires_grad=True,
            )
            for i in range(self._object_group_layer.num_obj)
        ]
        return pose_o

    def initialize_optimizer(self):
        # self._mse_loss = torch.nn.MSELoss(reduction="sum").to(self._device)
        self._meshsdf_loss = MeshSDFLoss().to(self._device)
        self._loss_reg_m = PoseAlignmentLoss(loss_type="l2_norm").to(self._device)
        self._loss_reg_o = PoseAlignmentLoss(loss_type="l2_norm").to(self._device)
        self._loss_smooth_m = PoseSmoothnessLoss(
            win_size=self._win_size_m,
            w_vel_r=self._w_smooth_rot_m,
            w_vel_t=self._w_smooth_trans_m,
            w_acc_r=self._w_smooth_acc_rot_m,
            w_acc_t=self._w_smooth_acc_trans_m,
        ).to(self._device)
        self._loss_smooth_o = PoseSmoothnessLoss(
            win_size=self._win_size_o,
            w_vel_r=self._w_smooth_rot_o,
            w_vel_t=self._w_smooth_trans_o,
            w_acc_r=self._w_smooth_acc_rot_o,
            w_acc_t=self._w_smooth_acc_trans_o,
        ).to(self._device)
        self._zero = torch.zeros((), dtype=torch.float32, device=self._device)

        poses_m = self._load_poses_m(self._pose_m_file)
        poses_o = self._load_poses_o(self._pose_o_file)

        self._pose_m = self._initialize_pose_m_from_poses(poses_m)
        self._pose_o = self._initialize_pose_o_from_poses(poses_o)
        self._optimizer = torch.optim.Adam(self._pose_o + self._pose_m, lr=self._lr)

        self._target_pose_m = torch.from_numpy(
            np.stack([p for p in poses_m], axis=0)
        ).to(self._device)
        self._target_pose_o = torch.from_numpy(
            np.stack([quat_to_rvt(p) for p in poses_o], axis=0)
        ).to(self._device)

    def _prepare_dpts_list_for_loss_sdf(self, verts, faces):
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
                    try:
                        dpts_list[idx] = torch.from_numpy(future.result()).to(
                            self._device
                        )
                    except Exception as e:
                        self._logger.error(f"Error in loading dpts: {e}")
                        dpts_list[idx] = torch.zeros(
                            (0, 3), dtype=torch.float32, device=self._device
                        )
                    finally:
                        tqbar.update(1)
                futures.clear()
            tqbar.close()
        else:
            self._logger.info(f"Generating dpts...")
            for f_idx in tqdm(range(self._num_frames), ncols=100):
                self._data_loader.step_by_frame_id(f_idx)
                points = self._data_loader.points[self._data_loader.masks]
                points = self._get_dpts_for_loss_sdf(
                    verts[f_idx], faces, points, self._dist_thresh
                )
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
                    f_idx = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self._logger.error(f"Error in saving dpts: {e}")
                    finally:
                        tqbar.update(1)
                futures.clear()
            tqbar.close()
        self._logger.info("Done preparing dpts for SDF loss.")
        return dpts_list

    def solve(self):
        subset_o = list(range(self._object_group_layer.num_obj))
        subset_m = list(range(self._mano_group_layer.num_obj))

        self._logger.info(">>>>>>>>>> Start optimization <<<<<<<<<<")
        t_s = time.time()

        self._log_loss = np.zeros(
            (6, self._total_steps), dtype=np.float32
        )  # total, sdf, reg_m, smooth_m, reg_o, smooth_o

        faces_o, _ = self._object_group_layer.get_f_from_inds(subset_o)
        faces_m, _ = self._mano_group_layer.get_f_from_inds(subset_m)
        faces = torch.cat(
            [
                faces_o,
                faces_m + self._object_group_layer.get_num_verts_from_inds(subset_o),
            ],
            dim=0,
        )

        # verts_o, _ = self._object_group_layer_forward(self._pose_o, subset_o)
        # verts_m, _ = self._mano_group_layer_forward(self._pose_m, subset_m)
        # verts = torch.cat([verts_o, verts_m], dim=1)

        # # Prepare dpts for SDF loss
        # dpts_list = self._prepare_dpts_list_for_loss_sdf(verts, faces)

        tt_s = time.time()
        for step in range(self._total_steps):
            ttt_s = time.time()

            self._optimizer.zero_grad()

            verts_o, _ = self._object_group_layer_forward(self._pose_o, subset_o)
            verts_m, _ = self._mano_group_layer_forward(self._pose_m, subset_m)
            verts = torch.cat([verts_o, verts_m], dim=1)

            if self._w_sdf > 0 and step == self._total_steps - self._sdf_steps:
                dpts_list = self._prepare_dpts_list_for_loss_sdf(verts, faces)
                tt_s = time.time()
                self._log_info_steps = 10
                self._log_debug_steps = 1

            # Compute losses
            if self._w_sdf == 0:
                loss_sdf = self._zero
            elif step >= self._total_steps - self._sdf_steps:
                loss_sdf = self._loss_sdf(verts, faces, dpts_list)
                loss_sdf *= self._w_sdf
            else:
                loss_sdf = self._zero

            if self._w_reg_m == 0:
                loss_reg_m = self._zero
            else:
                loss_reg_m = self._loss_reg_m(
                    self._pose_m, self._target_pose_m, subset_m
                )
                loss_reg_m *= self._w_reg_m

            if self._w_reg_o == 0:
                loss_reg_o = self._zero
            else:
                loss_reg_o = self._loss_reg_o(
                    self._pose_o, self._target_pose_o, subset_o
                )
                loss_reg_o *= self._w_reg_o

            if self._w_smooth_m == 0:
                loss_smooth_m = self._zero
            else:
                loss_smooth_m = self._loss_smooth_m(self._pose_m, subset_m)
                loss_smooth_m *= self._w_smooth_m

            if self._w_smooth_o == 0:
                loss_smooth_o = self._zero
            else:
                loss_smooth_o = self._loss_smooth_o(self._pose_o, subset_o)
                loss_smooth_o *= self._w_smooth_o

            loss = loss_sdf + loss_reg_m + loss_reg_o + loss_smooth_m + loss_smooth_o

            # Set grad to None to prevent Adam from updating the parameters even when
            # the grad are all zeros. An alternative is to place this before the
            # forward pass, but the currently implementation of group layers does not
            # allow this since the non-active poses will still always be used due to
            # concatenation and hence their grad will be 0.0 rather than None.
            for i, p in enumerate(self._pose_m):
                if i not in subset_m and p.grad is not None:
                    assert p.grad.sum() == 0.0
                    p.grad = None

            for i, p in enumerate(self._pose_o):
                if i not in subset_o and p.grad is not None:
                    assert p.grad.sum() == 0.0
                    p.grad = None

            loss.backward()
            self._optimizer.step()

            self._log_loss[:, step] = [
                loss.item(),
                loss_sdf.item(),
                loss_reg_m.item(),
                loss_smooth_m.item(),
                loss_reg_o.item(),
                loss_smooth_o.item(),
            ]

            log_msg = (
                f"step: {step+1:04d}/{self._total_steps:04d}"
                + f"| loss: {loss.item():11.8f} "
                + f"| sdf: {loss_sdf.item():11.8f} "
                + f"| reg_m: {loss_reg_m.item():11.8f} "
                + f"| smooth_m: {loss_smooth_m.item():11.8f} "
                + f"| reg_o: {loss_reg_o.item():11.8f} "
                + f"| smooth_o: {loss_smooth_o.item():11.8f}"
            )
            if (step + 1) % self._log_info_steps == 0:
                self._logger.info(log_msg + f"| time: {time.time() - tt_s:.2f}s")
                tt_s = time.time()
            elif (step + 1) % self._log_debug_steps == 0:
                self._logger.debug(log_msg + f"| time: {time.time() - ttt_s:.2f}s")

        self._logger.info(
            f">>>>>>>>>> Optimization Done! ({time.time() - t_s:.2f}s) <<<<<<<<<<"
        )

    def save_results(
        self, loss_name="loss", poses_m_name="poses_m", poses_o_name="poses_o"
    ):
        self._logger.info(">>>>>>>>>> Saving results <<<<<<<<<<")
        t_s = time.time()
        # Save loss log
        self._save_log_loss(loss_name)

        # Save optimized poses
        self._save_optimized_poses_m(poses_m_name)
        self._save_optimized_poses_o(poses_o_name)
        self._logger.info(
            f">>>>>>>>>> Saving results Done!!! ({time.time() - t_s:.2f}s) <<<<<<<<<<"
        )

    def render_optimized_poses(
        self,
        video_name="vis_joint_pose_solver",
        poses_m_name="poses_m",
        poses_o_name="poses_o",
    ):
        self._logger.info("Rendering optimized poses...")
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

        # Prepare object data
        poses_o = np.load(self._save_folder / f"{poses_o_name}.npy").astype(np.float32)
        poses_o = np.stack([quat_to_mat(p) for p in poses_o], axis=1)
        self._logger.debug(f"Loaded poses_o: {poses_o.shape}")

        # Render images
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
        self._logger.info("Start Joint Pose Solver")
        self._logger.info("=" * 100)
        t_s = time.time()

        # Initialize optimizer
        self.initialize_optimizer()

        # Start optimization
        self.solve()

        # Save results
        self.save_results()

        # Render optimized poses
        self.render_optimized_poses()

        self._logger.info("=" * 100)
        self._logger.info(f"Joint Pose Solver Done!!! ({time.time() - t_s:.2f}s)")
        self._logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Pose Solver")
    parser.add_argument(
        "--sequence_folder",
        type=str,
        default=None,
        help="Path to the sequence folder.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the solver in debug mode."
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder path.")

    solver = JointPoseSolver(args.sequence_folder, args.debug)
    solver.run()
