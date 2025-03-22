import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU-based offscreen rendering
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from hocap_annotation.utils import *
from pprint import pformat
from hocap_annotation.loss import (
    MeshSDFLoss,
    Keypoint2DLoss,
    Keypoint3DLoss,
    MANORegLoss,
    PoseAlignmentLoss,
    PoseSmoothnessLoss,
)
from hocap_annotation.loaders import SequenceLoader
from hocap_annotation.rendering import HOCapRenderer


class ObjectPoseSolver:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._data_folder = Path(sequence_folder)
        self._debug = debug
        self._device = CFG.device
        self._save_folder = self._data_folder / "processed" / "object_pose_solver"
        self._save_folder.mkdir(parents=True, exist_ok=True)

        self._log_file = self._save_folder / "object_pose_solver.log"
        # Remove the existing log file
        if self._log_file.exists():
            self._log_file.unlink()
        self._logger = get_logger(
            self.__class__.__name__, "DEBUG" if debug else "INFO", self._log_file
        )

        self._log_info_steps = 10
        self._log_debug_steps = 1

        # Check if the required files exist
        self._check_required_files()

        # Load optimization config
        self._load_optim_config()

        # Load parameters from data loader
        self._load_dataloader_params()

    def _load_optim_config(self):
        self._logger.info("Loading optimization configuration...")
        optim_config = CFG.optimization.object_pose_solver
        self._lr = optim_config["lr"]
        self._total_steps = optim_config["total_steps"]
        self._sdf_steps = optim_config["sdf_steps"]
        self._smooth_steps = optim_config["smooth_steps"]
        self._w_sdf = optim_config["w_sdf"]
        self._w_reg = optim_config["w_reg"]
        self._w_smooth = optim_config["w_smooth"]
        self._w_smooth_rot = optim_config["w_smooth_rot"]
        self._w_smooth_trans = optim_config["w_smooth_trans"]
        self._w_smooth_acc_rot = optim_config["w_smooth_acc_rot"]
        self._w_smooth_acc_trans = optim_config["w_smooth_acc_trans"]
        self._win_size = optim_config["smooth_window_size"]
        self._dist_thresh = optim_config["sdf_dist_thresh"]
        self._use_object_masks = optim_config["use_object_masks"]
        self._load_offline_dpts = optim_config["load_offline_dpts"]
        self._logger.debug(
            "Optimization Config:\n" + pformat(optim_config, sort_dicts=False)
        )

    def _check_required_files(self):
        self._logger.info("Checking existence of required files...")
        fd_pose_folder = self._data_folder / "processed/fd_pose_solver"
        poses_o_files = [self._data_folder / "poses_o.npy"]
        for f in poses_o_files:
            if f.exists():
                self._fd_pose_file = f
                self._logger.info(f"Object poses file: {f}")
                return
        else:
            msg = "Object poses not found!"
            self._logger.error(msg)
            raise FileNotFoundError(msg)

    def _load_dataloader_params(self):
        self._data_loader = SequenceLoader(
            self._data_folder, load_object=True, device=self._device
        )
        self._num_frames = self._data_loader.num_frames
        self._rs_serials = self._data_loader.rs_serials
        self._object_group_layer = self._data_loader.object_group_layer

    def _load_poses_o(self, pose_file):
        poses = np.load(pose_file).astype(np.float32)
        self._logger.info(f"Object poses loaded: {poses.shape}")
        return poses

    def _object_group_layer_forward(self, pose_o, subset=None):
        p = torch.cat(pose_o, dim=1)
        v, n = self._object_group_layer(p, subset)
        if v.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

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

    def _prepare_dpts_list_for_loss_sdf(self, verts, faces, subset):
        self._logger.info("Preparing dpts for SDF loss...")

        save_dpts_folder = self._save_folder / "dpts"
        save_dpts_folder.mkdir(parents=True, exist_ok=True)

        dpts_files = sorted(save_dpts_folder.glob("dpts_*.ply"))
        dpts_list = [None] * self._num_frames
        if self._load_offline_dpts and len(dpts_files) == self._num_frames:
            self._logger.info("Loading offline dpts...")
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
                pcd_points = self._data_loader.points
                pcd_masks = self._data_loader.masks

                if self._use_object_masks:
                    seg_masks = [
                        erode_mask(self._data_loader.get_mask_image(f_idx, s), 3)
                        for s in self._rs_serials
                    ]
                    seg_masks = np.stack(seg_masks, axis=0).reshape(
                        len(self._rs_serials), -1
                    )
                    masks_o = np.isin(seg_masks, np.array(subset) + 1)
                    masks_o = torch.from_numpy(masks_o).to(self._device)
                    pcd_masks = torch.logical_and(pcd_masks, masks_o)

                pcd_points = pcd_points[pcd_masks]

                if not self._use_object_masks:
                    pcd_points = self._get_dpts_for_loss_sdf(
                        verts[f_idx], faces, pcd_points, self._dist_thresh
                    )

                pcd_points = process_points(
                    points=pcd_points, voxel_size=0.003, remove_outliers=True
                )

                if pcd_points.size(0) == 0:
                    self._logger.warning(
                        f"No valid dpts for frame {f_idx}, using zeros."
                    )
                    pcd_points = torch.zeros(
                        (0, 3), dtype=torch.float32, device=self._device
                    )

                dpts_list[f_idx] = pcd_points

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

    def _initialize_pose_o_from_poses_o(self, poses_o):
        pose_o = [
            torch.nn.Parameter(
                torch.from_numpy(quat_to_rvt(poses_o[i])).to(self._device),
                requires_grad=True,
            )
            for i in range(self._object_group_layer.num_obj)
        ]
        return pose_o

    def _save_log_loss(self, save_name):
        self._logger.info("Saving loss log...")
        np.savez(
            self._save_folder / f"{save_name}.npz",
            total=self._log_loss[0],
            sdf=self._log_loss[1],
            reg=self._log_loss[2],
            smooth=self._log_loss[3],
        )
        loss_curve_img = draw_losses_curve(
            self._log_loss, ["total", "sdf", "reg", "smooth"]
        )
        write_rgb_image(self._save_folder / f"{save_name}_curve.png", loss_curve_img)

    def _save_optimized_poses_o(self, save_name):
        self._logger.info("Saving optimized object poses...")
        optim_pose_o = torch.stack([p.data for p in self._pose_o], dim=0).squeeze(0)
        optim_pose_o = optim_pose_o.cpu().numpy().astype(np.float32)
        optim_pose_o = np.stack(
            [rvt_to_quat(ps) for ps in optim_pose_o], dtype=np.float32
        )
        self._logger.debug(f"optim_pose_o: {optim_pose_o.shape}")
        # np.save(self._save_folder / f"{save_name}_raw.npy", optim_pose_o)

        # # Smooth the poses
        # self._logger.info("Smoothing optimized poses...")
        # for i in range(len(optim_pose_o)):
        #     optim_pose_o[i] = evaluate_and_fix_poses(
        #         optim_pose_o[i],
        #         window_size=15,
        #         rot_thresh=1.0,
        #         trans_thresh=0.001,
        #         seperate_rot_trans=False,
        #     )
        #     optim_pose_o[i] = evaluate_and_fix_poses(
        #         optim_pose_o[i],
        #         window_size=30,
        #         rot_thresh=0.1,
        #         trans_thresh=0.01,
        #         seperate_rot_trans=False,
        #     )
        np.save(self._save_folder / f"{save_name}.npy", optim_pose_o)

    def initialize_optimizer(self):
        self._logger.info(">>>>>>>>>> Initializing optimizer <<<<<<<<<<")
        # self._mse_loss = torch.nn.MSELoss(reduction="sum").to(self._device)
        self._meshsdf_loss = MeshSDFLoss().to(self._device)
        self._loss_reg_o = PoseAlignmentLoss(loss_type="l2_norm").to(self._device)
        self._loss_smooth = PoseSmoothnessLoss(
            win_size=self._win_size,
            w_vel_r=self._w_smooth_rot,
            w_vel_t=self._w_smooth_trans,
            w_acc_r=self._w_smooth_acc_rot,
            w_acc_t=self._w_smooth_acc_trans,
        ).to(self._device)
        self._zero = torch.zeros((), dtype=torch.float32, device=self._device)

        poses_o = self._load_poses_o(self._fd_pose_file)

        self._pose_o = self._initialize_pose_o_from_poses_o(poses_o)
        self._optimizer = torch.optim.Adam(self._pose_o, lr=self._lr)

        self._target_pose_o = torch.from_numpy(
            np.stack([quat_to_rvt(p) for p in poses_o], axis=0)
        ).to(self._device)
        self._logger.debug(f"target_pose_o: {self._target_pose_o.shape}")

        self._log_loss = np.zeros((4, self._total_steps), dtype=np.float32)

    def solve(self):
        subset_o = list(range(self._object_group_layer.num_obj))

        self._logger.info(">>>>>>>>>> Start optimization <<<<<<<<<<")
        t_s = time.time()

        faces_o, _ = self._object_group_layer.get_f_from_inds(subset_o)

        tt_s = time.time()
        ttt_s = time.time()
        for step in range(self._total_steps):
            self._optimizer.zero_grad()

            verts_o, _ = self._object_group_layer_forward(self._pose_o, subset_o)

            # Prepare dpts for SDF loss
            if self._w_sdf > 0 and step == self._total_steps - self._sdf_steps:
                dpts_list = self._prepare_dpts_list_for_loss_sdf(
                    verts_o, faces_o, subset_o
                )
                tt_s = time.time()
                ttt_s = time.time()
                self._log_info_steps = 10
                self._log_debug_steps = 1

            # Calculate losses
            if self._w_sdf == 0:
                loss_sdf = self._zero
            elif step >= self._total_steps - self._sdf_steps:
                loss_sdf = self._loss_sdf(verts_o, faces_o, dpts_list)
                loss_sdf *= self._w_sdf
            else:
                loss_sdf = self._zero

            if self._w_reg == 0:
                loss_reg = self._zero
            else:
                loss_reg = self._loss_reg_o(self._pose_o, self._target_pose_o, subset_o)
                loss_reg *= self._w_reg

            if self._w_smooth == 0:
                loss_smooth = self._zero
            elif step >= self._total_steps - self._smooth_steps:
                loss_smooth = self._loss_smooth(self._pose_o, subset_o)
                loss_smooth *= self._w_smooth
            else:
                loss_smooth = self._zero

            loss = loss_sdf + loss_reg + loss_smooth

            # Set grad to None to prevent Adam from updating the parameters even when
            # the grad are all zeros. An alternative is to place this before the
            # forward pass, but the currently implementation of group layers does not
            # allow this since the non-active poses will still always be used due to
            # concatenation and hence their grad will be 0.0 rather than None.
            for i, p in enumerate(self._pose_o):
                if i not in subset_o and p.grad is not None:
                    assert p.grad.sum() == 0.0
                    p.grad = None

            loss.backward()
            self._optimizer.step()

            self._log_loss[:, step] = [
                loss.item(),
                loss_sdf.item(),
                loss_reg.item(),
                loss_smooth.item(),
            ]

            log_msg = (
                f"step: {step+1:04d}/{self._total_steps:04d}"
                + f"| loss: {loss.item():11.8f} "
                + f"| sdf: {loss_sdf.item():11.8f} "
                + f"| reg: {loss_reg.item():11.8f} "
                + f"| smooth: {loss_smooth.item():11.8f} "
            )
            if (step + 1) % self._log_info_steps == 0:
                self._logger.info(log_msg + f"| time: {time.time() - tt_s:.2f}s")
                tt_s = time.time()
            elif (step + 1) % self._log_debug_steps == 0:
                self._logger.debug(log_msg + f"| time: {time.time() - ttt_s:.2f}s")
                ttt_s = time.time()

        self._logger.info(f"Optimization Done!!! ({time.time() - t_s:.2f}s)")

    def save_results(self, loss_name="loss", poses_o_name="poses_o"):
        self._logger.info(">>>>>>>>>> Saving results <<<<<<<<<<")
        t_s = time.time()

        # Save loss log
        self._save_log_loss(loss_name)

        # Save optimized poses
        self._save_optimized_poses_o(poses_o_name)

        self._logger.info(
            f">>>>>>>>>> Saving results Done!!! ({time.time() - t_s:.2f}s) <<<<<<<<<<"
        )

    def render_optimized_poses(
        self, video_name="vis_object_pose_solver", poses_o_name="poses_o"
    ):
        self._logger.info("Rendering optimized poses...")
        poses_o, verts_m, faces_m, colors_m, joints_m = None, None, None, None, None

        # Prepare object data
        poses_o = self._load_poses_o(self._save_folder / f"{poses_o_name}.npy")
        poses_o = np.stack([quat_to_mat(p) for p in poses_o], axis=1)
        self._logger.debug(f"Loaded poses_o: {poses_o.shape}")

        # Render poses
        renderer = HOCapRenderer(self._data_folder, log_file=self._log_file)
        renderer.update_render_dict(poses_o, verts_m, faces_m, colors_m, joints_m)
        renderer.render_pose_images(
            save_folder=self._save_folder / "vis",
            save_video_path=self._save_folder / f"{video_name}.mp4",
            vis_only=True,
            save_vis=True,
        )

    def run(self):
        self._logger.info("=" * 100)
        self._logger.info("Start Object Pose Solver")
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
        self._logger.info(f"Object Pose Solver Done!!! ({time.time() - t_s:.2f}s)")
        self._logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Pose Solver")
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

    solver = ObjectPoseSolver(args.sequence_folder, args.debug)
    solver.run()
