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
from hocap_annotation.loaders import MySequenceLoader as SequenceLoader
from hocap_annotation.rendering import HOCapRenderer
import pickle
from manopth.manolayer import ManoLayer
from hocap_annotation.utils import CFG
from tqdm import tqdm
from torch.nn import Parameter
from torch.optim import Adam
import torch
from scipy.spatial.transform import Rotation as R
import concurrent.futures


def load_pkl_and_get_hand_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if 'hand_pose' not in data:
        raise ValueError("No 'hand_pose' found in the .pkl file.")
    hand_pose = data['hand_pose']
    left_hand_pose = np.array(hand_pose.get('left_hand_pose', []))
    left_hand_beta = np.array(hand_pose.get('left_hand_beta', []))
    left_hand_translation = np.array(hand_pose.get('left_hand_translation', []))
    left_hand_base_rot = np.array(hand_pose.get('left_hand_base_rot', []))
    right_hand_pose = np.array(hand_pose.get('right_hand_pose', []))
    right_hand_beta = np.array(hand_pose.get('right_hand_beta', []))
    right_hand_translation = np.array(hand_pose.get('right_hand_translation', []))
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

def reconstruct_left_hand_mesh(hand_data, frame_idx, mano_layer_right, device, mano_layer_left):
    """
    Reconstruct left hand mesh using the RIGHT MANO layer (intentional, not a mistake).
    This matches the pattern used in visualize_wilor_hand_video.py.
    """
    pose = torch.tensor(hand_data['left_hand_pose'][frame_idx]).to(device).unsqueeze(0)
    translation = torch.tensor(hand_data['left_hand_translation'][frame_idx]).to(device).unsqueeze(0)
    base_rot = torch.tensor(hand_data['left_hand_base_rot'][frame_idx]).to(device) if hand_data['left_hand_base_rot'].ndim == 3 else torch.eye(3).to(device)
    hand_beta = torch.tensor(hand_data['left_hand_beta']).to(device)
    
    # Use the RIGHT layer for left hand (intentional)
    verts, joints = mano_layer_right(pose, hand_beta.float())
    verts = verts[0] / 1000  # Convert from mm to meters
    joints = joints[0] / 1000
    
    if verts.size(0) == 1:
        verts = verts.squeeze(0)
        joints = joints.squeeze(0)
    
    root_trans = joints[0].clone().detach()
    verts -= root_trans
    verts[:, 0] *= -1
    verts = verts @ base_rot.T
    verts += translation
    
    # Get faces from the RIGHT layer (since we used it for reconstruction)
    faces = mano_layer_left.th_faces.detach().cpu().numpy()
    return verts.detach().cpu().numpy(), faces

def reconstruct_right_hand_mesh(hand_data, frame_idx, mano_layer_right, device):
    """
    Reconstruct right hand mesh using the RIGHT MANO layer.
    This matches the pattern used in visualize_wilor_hand_video.py.
    """
    pose = torch.tensor(hand_data['right_hand_pose'][frame_idx]).to(device).unsqueeze(0)
    translation = torch.tensor(hand_data['right_hand_translation'][frame_idx]).to(device).unsqueeze(0)
    hand_beta = torch.tensor(hand_data['right_hand_beta']).to(device)  # Use left hand beta for both hands
    
    # Use the RIGHT layer for right hand
    verts, joints = mano_layer_right(pose, hand_beta.float())
    verts = verts[0] / 1000  # Convert from mm to meters
    joints = joints[0] / 1000
    
    if verts.size(0) == 1:
        verts = verts.squeeze(0)
        joints = joints.squeeze(0)
    
    root_trans = joints[0].clone().detach()
    verts -= root_trans
    verts += translation
    
    # Get faces from the RIGHT layer
    faces = mano_layer_right.th_faces.detach().cpu().numpy()
    return verts.detach().cpu().numpy(), faces


class JointPoseSolver:
    def __init__(self, sequence_folder, debug=False) -> None:
        self._data_folder = Path(sequence_folder)
        self._debug = debug
        self._device = CFG.device
        self._folder_name = self._data_folder.parent.parent.name
        self._task_name = self._data_folder.parent.name
        self._sequence_name = self._data_folder.name
        self._save_folder = Path(f"{self._data_folder.parent.parent.parent}/{self._folder_name}_annotated/{self._task_name}/{self._sequence_name}/processed/joint_pose_solver")
        self._save_folder.mkdir(parents=True, exist_ok=True)
        self._annotated_folder = Path(f"{self._data_folder.parent.parent.parent}/{self._folder_name}_annotated/{self._task_name}/{self._sequence_name}")

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
            # self._data_folder / "processed" / "object_pose_solver" / "poses_o.npy"
            self._annotated_folder / "processed" / "object_pose_solver" / "poses_o.npy"
            
        )
        self._pose_m_file = (
            # self._data_folder / "processed" / "hand_pose_solver" / "poses_m.npy"
            # self._data_folder / "processed" /  "poses_m.npy"
            self._annotated_folder  /  "poses_m.npy"
        )
        self._hand_pkl_file = (
            # self._data_folder / "processed" / "result_hand_optimized.pkl"
            self._annotated_folder / "result_hand_optimized.pkl"
        )
        msg = "File not found: {}"
        if not self._pose_o_file.exists():
            self._logger.error(msg.format(self._pose_o_file))
            raise FileNotFoundError(msg.format(self._pose_o_file))
        else:
            self._logger.info(f"Object poses file: {self._pose_o_file}")

        if not self._hand_pkl_file.exists():
            self._logger.error(msg.format(self._hand_pkl_file))
            raise FileNotFoundError(msg.format(self._hand_pkl_file))
        else:
            self._logger.info(f"Hand poses pickle file: {self._hand_pkl_file}")
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

        # --- Load hand pkl and set up MANOLayers for left/right hand ---
        self._hand_data = load_pkl_and_get_hand_data(self._hand_pkl_file)
        self._mano_layer_left = ManoLayer(side="left",
                                mano_root=CFG.mano.model_path, 
                                use_pca=False, 
                                ncomps=45).to('cuda').to(self._device)
        self._mano_layer_right = ManoLayer(side="right",
                                mano_root=CFG.mano.model_path, 
                                use_pca=False, 
                                ncomps=45).to('cuda').to(self._device)

    def _load_poses_o(self, pose_file):
        poses = np.load(pose_file).astype(np.float32)
        if poses.ndim != 3:
            poses = np.expand_dims(poses, axis=0)
        self._logger.debug(f"Object poses loaded: {poses.shape}")
        return poses

    def _load_poses_m(self, pose_file):
        # Load hand poses from pickle file instead of numpy file
        hand_data = load_pkl_and_get_hand_data(self._hand_pkl_file)
        
        # Extract poses for left and right hands
        left_poses = hand_data['left_hand_pose']  # Shape: (num_frames, 51)
        right_poses = hand_data['right_hand_pose']  # Shape: (num_frames, 51)
        
        # Stack poses for both hands: (2, num_frames, 51)
        poses = np.stack([left_poses, right_poses], axis=0)
        
        # Filter poses based on available mano sides
        poses = np.stack(
            [poses[0 if side == "right" else 1] for side in self._mano_sides], axis=0
        )  # (num_hands, num_frames, 51)
        
        self._logger.info(f"Hand poses loaded from pickle: {poses.shape}")
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

    def _mano_layer_forward_separate(self, pose_m, frame_idx=None):
        """
        Forward left and right hand separately using their respective MANO layers.
        This follows the pattern from visualize_wilor_hand_video.py:
        - Left hand uses RIGHT MANO layer (intentional, not a mistake)
        - Right hand uses RIGHT MANO layer
        - Both hands use left_hand_beta
        pose_m: list of [left_pose, right_pose], each shape (1, 51)
        frame_idx: int, current frame index (to get translation from self._hand_data)
        Returns: verts (N_verts, 3), faces (N_faces, 3)
        """
        import torch
        device = self._device
        if frame_idx is None:
            raise ValueError("frame_idx must be provided to get translation for each hand.")
        
        # Use optimizable translation Parameters (gradients flow through these)
        def _get_left_trans(fidx):
            if self._left_hand_trans is not None:
                return self._left_hand_trans[fidx].unsqueeze(0)
            return torch.tensor(self._hand_data['left_hand_translation'][fidx]).to(device).float().unsqueeze(0)

        def _get_right_trans(fidx):
            if self._right_hand_trans is not None:
                return self._right_hand_trans[fidx].unsqueeze(0)
            return torch.tensor(self._hand_data['right_hand_translation'][fidx]).to(device).float().unsqueeze(0)

        # Handle case where we might have only one hand
        if len(pose_m) == 1:
            # Only one hand available
            if "right" in self._mano_sides:
                # Right hand only
                right_pose = pose_m[0]  # (1, 51)
                right_translation = _get_right_trans(frame_idx)
                right_beta = torch.tensor(self._hand_data['left_hand_beta']).to(device).float()

                verts_right, joints_right = self._mano_layer_right(right_pose, right_beta)
                verts_right = verts_right[0] / 1000
                joints_right = joints_right[0] / 1000

                root_trans_right = joints_right[0].clone().detach()
                verts_right -= root_trans_right
                verts_right += right_translation

                verts = verts_right
                faces = self._mano_layer_right.th_faces.detach().clone()
                return verts, faces
            else:
                # Left hand only
                left_pose = pose_m[0]  # (1, 51)
                left_translation = _get_left_trans(frame_idx)
                left_beta = torch.tensor(self._hand_data['left_hand_beta']).to(device).float()

                verts_left, joints_left = self._mano_layer_right(left_pose, left_beta)
                verts_left = verts_left[0] / 1000
                joints_left = joints_left[0] / 1000

                root_trans_left = joints_left[0].clone().detach()
                verts_left -= root_trans_left
                verts_left[:, 0] *= -1

                if self._hand_data['left_hand_base_rot'].ndim == 3:
                    rot_idx = min(frame_idx, self._hand_data['left_hand_base_rot'].shape[0] - 1)
                    base_rot = torch.tensor(self._hand_data['left_hand_base_rot'][rot_idx]).to(device).float()
                    verts_left = verts_left @ base_rot.T

                verts_left += left_translation
                verts = verts_left
                faces = self._mano_layer_left.th_faces.detach().cpu().numpy()
                return verts, faces

        # Both hands available
        left_pose = pose_m[0]  # (1, 51)
        right_pose = pose_m[1]  # (1, 51)
        left_translation = _get_left_trans(frame_idx)
        right_translation = _get_right_trans(frame_idx)

        # Both hands use left_hand_beta (following visualize_wilor_hand_video.py pattern)
        left_beta = torch.tensor(self._hand_data['left_hand_beta']).to(device).float()
        right_beta = torch.tensor(self._hand_data['left_hand_beta']).to(device).float()

        # Left hand uses RIGHT MANO layer (intentional, not a mistake)
        verts_left, joints_left = self._mano_layer_right(left_pose, left_beta)
        verts_right, joints_right = self._mano_layer_right(right_pose, right_beta)

        # Convert to meters (mm to m)
        verts_left = verts_left[0] / 1000
        verts_right = verts_right[0] / 1000
        joints_left = joints_left[0] / 1000
        joints_right = joints_right[0] / 1000

        # Apply root translation offset
        root_trans_left = joints_left[0].clone().detach()
        root_trans_right = joints_right[0].clone().detach()
        verts_left -= root_trans_left
        verts_right -= root_trans_right

        # Apply left hand specific transformations (mirroring and rotation)
        verts_left[:, 0] *= -1
        if self._hand_data['left_hand_base_rot'].ndim == 3:
            rot_idx = min(frame_idx, self._hand_data['left_hand_base_rot'].shape[0] - 1)
            base_rot = torch.tensor(self._hand_data['left_hand_base_rot'][rot_idx]).to(device).float()
            verts_left = verts_left @ base_rot.T

        # Apply translations
        verts_left += left_translation
        verts_right += right_translation

        if verts_left.size(0) == 1:
            verts_left = verts_left.squeeze(0)
        if verts_right.size(0) == 1:
            verts_right = verts_right.squeeze(0)
        verts = torch.cat([verts_left, verts_right], dim=0)

        # Get faces from RIGHT MANO layer (since both hands use it)
        faces_left = self._mano_layer_right.th_faces.detach().clone()
        faces_right = self._mano_layer_right.th_faces.detach().clone() + verts_left.shape[0]
        faces = torch.cat([faces_left, faces_right], dim=0)
        return verts, faces

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
        
        # Ensure we have both left and right hand poses
        if optim_pose_m.shape[0] == 1:
            # If only one hand, create filler for the other
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
        # poses_m shape: (num_hands, num_frames, 51)
        pose_m = [
            Parameter(
                torch.from_numpy(poses_m[i]).to(self._device), requires_grad=True
            )
            for i in range(poses_m.shape[0])  # Use actual number of hands from poses_m
        ]
        return pose_m

    def _initialize_pose_o_from_poses(self, poses_o):
        pose_o = [
            Parameter(
                torch.from_numpy(quat_to_rvt(poses_o[i])).to(self._device),
                requires_grad=True,
            )
            for i in range(self._object_group_layer.num_obj)
        ]
        return pose_o

    def _initialize_hand_translations(self):
        """Make hand translations optimizable Parameters."""
        trans_params = []
        # Left hand translation
        if len(self._hand_data['left_hand_translation']) > 0:
            left_trans = torch.tensor(
                self._hand_data['left_hand_translation'], dtype=torch.float32
            ).to(self._device)
            self._left_hand_trans = Parameter(left_trans, requires_grad=True)
            trans_params.append(self._left_hand_trans)
            self._target_left_trans = left_trans.clone().detach()
        else:
            self._left_hand_trans = None
            self._target_left_trans = None

        # Right hand translation
        if len(self._hand_data['right_hand_translation']) > 0:
            right_trans = torch.tensor(
                self._hand_data['right_hand_translation'], dtype=torch.float32
            ).to(self._device)
            self._right_hand_trans = Parameter(right_trans, requires_grad=True)
            trans_params.append(self._right_hand_trans)
            self._target_right_trans = right_trans.clone().detach()
        else:
            self._right_hand_trans = None
            self._target_right_trans = None

        return trans_params

    def initialize_optimizer(self):
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
        trans_params = self._initialize_hand_translations()
        self._logger.info(f"Optimizable params: {len(self._pose_m)} hand poses, "
                          f"{len(self._pose_o)} object poses, {len(trans_params)} hand translations")
        self._optimizer = Adam(self._pose_o + self._pose_m + trans_params, lr=self._lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=self._total_steps, eta_min=self._lr * 0.01
        )

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
                # Use the verts for this specific frame
                verts_frame = verts[f_idx]
                points = self._get_dpts_for_loss_sdf(
                    verts_frame, faces, points, self._dist_thresh
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
        subset_m = list(range(len(self._pose_m)))  # Use actual number of hand poses

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

        # Prepare dpts at the start (before optimization changes poses)
        if self._w_sdf > 0:
            verts_list = []
            for f_idx in range(self._num_frames):
                if len(self._pose_m) == 2:
                    lp = self._pose_m[0][f_idx].unsqueeze(0)
                    rp = self._pose_m[1][f_idx].unsqueeze(0)
                    verts_m_f, faces_m_f = self._mano_layer_forward_separate([lp, rp], frame_idx=f_idx)
                else:
                    sp = self._pose_m[0][f_idx].unsqueeze(0)
                    verts_m_f, faces_m_f = self._mano_layer_forward_separate([sp], frame_idx=f_idx)
                verts_o_f, _ = self._object_group_layer_forward(self._pose_o, subset_o)
                verts_f = torch.cat([verts_o_f[f_idx], verts_m_f], dim=0)
                verts_list.append(verts_f.detach())

            faces_for_dpts = torch.cat([
                faces_o,
                faces_m_f + self._object_group_layer.get_num_verts_from_inds(subset_o)
            ], dim=0)
            dpts_list = self._prepare_dpts_list_for_loss_sdf(verts_list, faces_for_dpts)

        for step in range(self._total_steps):
            ttt_s = time.time()

            self._optimizer.zero_grad()

            # Accumulate loss over ALL frames
            loss_sdf_accum = self._zero
            verts_o_all, _ = self._object_group_layer_forward(self._pose_o, subset_o)

            for frame_idx in range(self._num_frames):
                if len(self._pose_m) == 2:
                    left_pose = self._pose_m[0][frame_idx].unsqueeze(0)
                    right_pose = self._pose_m[1][frame_idx].unsqueeze(0)
                    verts_m, faces_m = self._mano_layer_forward_separate([left_pose, right_pose], frame_idx=frame_idx)
                else:
                    single_pose = self._pose_m[0][frame_idx].unsqueeze(0)
                    verts_m, faces_m = self._mano_layer_forward_separate([single_pose], frame_idx=frame_idx)

                verts_frame = torch.cat([verts_o_all[frame_idx], verts_m], dim=0)
                faces = torch.cat([
                    faces_o,
                    faces_m + self._object_group_layer.get_num_verts_from_inds(subset_o)
                ], dim=0)

                # SDF loss per frame
                if self._w_sdf > 0 and step >= self._total_steps - self._sdf_steps:
                    current_dpts = dpts_list[frame_idx]
                    frame_sdf = self._loss_sdf([verts_frame], faces, [current_dpts])
                    loss_sdf_accum = loss_sdf_accum + frame_sdf

            # Average SDF over frames
            if self._w_sdf > 0 and step >= self._total_steps - self._sdf_steps:
                loss_sdf = (loss_sdf_accum / self._num_frames) * self._w_sdf
            else:
                loss_sdf = self._zero

            if self._w_reg_m == 0:
                loss_reg_m = self._zero
            else:
                loss_reg_m = self._loss_reg_m(
                    self._pose_m, self._target_pose_m, subset_m
                )
                # Add translation regularization (prevent hand from drifting too far)
                if self._left_hand_trans is not None and self._target_left_trans is not None:
                    loss_reg_m = loss_reg_m + torch.nn.functional.mse_loss(
                        self._left_hand_trans, self._target_left_trans
                    )
                if self._right_hand_trans is not None and self._target_right_trans is not None:
                    loss_reg_m = loss_reg_m + torch.nn.functional.mse_loss(
                        self._right_hand_trans, self._target_right_trans
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
            self._scheduler.step()

            self._log_loss[:, step] = [
                loss.item(),
                loss_sdf.item(),
                loss_reg_m.item(),
                loss_smooth_m.item(),
                loss_reg_o.item(),
                loss_smooth_o.item(),
            ]

            cur_lr = self._scheduler.get_last_lr()[0]
            log_msg = (
                f"step: {step+1:04d}/{self._total_steps:04d}"
                + f"| loss: {loss.item():11.8f} "
                + f"| sdf: {loss_sdf.item():11.8f} "
                + f"| reg_m: {loss_reg_m.item():11.8f} "
                + f"| smooth_m: {loss_smooth_m.item():11.8f} "
                + f"| reg_o: {loss_reg_o.item():11.8f} "
                + f"| smooth_o: {loss_smooth_o.item():11.8f}"
                + f"| lr: {cur_lr:.6f}"
            )
            if (step + 1) % self._log_info_steps == 0:
                self._logger.info(log_msg + f"| time: {time.time() - tt_s:.2f}s")
                tt_s = time.time()
            elif (step + 1) % self._log_debug_steps == 0:
                self._logger.debug(log_msg + f"| time: {time.time() - ttt_s:.2f}s")

        self._logger.info(
            f">>>>>>>>>> Optimization Done! ({time.time() - t_s:.2f}s) <<<<<<<<<<"
        )

    def _save_optimized_translations(self):
        """Save optimized hand translations back to pickle."""
        self._logger.info("Saving optimized hand translations...")
        pkl_path = self._hand_pkl_file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if self._left_hand_trans is not None:
            data['hand_pose']['left_hand_translation'] = self._left_hand_trans.detach().cpu().numpy().astype(np.float32)
        if self._right_hand_trans is not None:
            data['hand_pose']['right_hand_translation'] = self._right_hand_trans.detach().cpu().numpy().astype(np.float32)

        save_pkl_path = self._save_folder / "result_hand_optimized.pkl"
        with open(save_pkl_path, 'wb') as f:
            pickle.dump(data, f)
        self._logger.info(f"Saved optimized pkl to {save_pkl_path}")

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
        self._save_optimized_translations()
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
        # --- Hand mesh generation for visualization ---
        num_frames = self._num_frames
        left_hand_meshes = []
        right_hand_meshes = []
        for i in range(num_frames):
            # Following visualize_wilor_hand_video.py pattern:
            # Left hand uses RIGHT MANO layer (intentional, not a mistake)
            # Right hand uses RIGHT MANO layer
            if "left" in self._mano_sides:
                verts_left, faces_left = reconstruct_left_hand_mesh(self._hand_data, i, self._mano_layer_right, self._device, self._mano_layer_left)
                left_hand_meshes.append((verts_left, faces_left))
            else:
                left_hand_meshes.append((np.zeros((0, 3)), np.zeros((0, 3))))
                
            if "right" in self._mano_sides:
                verts_right, faces_right = reconstruct_right_hand_mesh(self._hand_data, i, self._mano_layer_right, self._device)
                right_hand_meshes.append((verts_right, faces_right))
            else:
                right_hand_meshes.append((np.zeros((0, 3)), np.zeros((0, 3))))
                
        self._logger.info(f"Generated {len(left_hand_meshes)} left and {len(right_hand_meshes)} right hand meshes from pkl.")

        # --- Combine left and right hand meshes for verts_m, faces_m, joints_m ---
        verts_m = []
        faces_m = []
        joints_m = []
        for i in range(num_frames):
            verts_left, faces_left = left_hand_meshes[i]
            verts_right, faces_right = right_hand_meshes[i]
            
            # Check if meshes are empty
            if verts_left.shape[0] == 0 and verts_right.shape[0] == 0:
                # No hands available
                verts_combined = np.zeros((0, 3))
                faces_combined = np.zeros((0, 3))
            elif verts_left.shape[0] == 0:
                # Only right hand
                verts_combined = verts_right
                faces_combined = faces_right
            elif verts_right.shape[0] == 0:
                # Only left hand
                verts_combined = verts_left
                faces_combined = faces_left
            else:
                # Both hands available
                verts_combined = np.concatenate([verts_left, verts_right], axis=0)
                faces_right_offset = faces_right + verts_left.shape[0]
                faces_combined = np.concatenate([faces_left, faces_right_offset], axis=0)
            
            verts_m.append(verts_combined)
            faces_m.append(faces_combined)
            # If you want to combine joints, you can extract them similarly (if available)
            # For now, just append None as placeholder
            joints_m.append(None)
        # Now verts_m, faces_m, joints_m are per-frame lists
        # You can use verts_m[frame_idx], faces_m[frame_idx], joints_m[frame_idx] in your renderer or optimizer
        # Prepare object data
        poses_o = np.load(self._save_folder / f"{poses_o_name}.npy").astype(np.float32)
        if poses_o.ndim != 3:
            poses_o = np.expand_dims(poses_o, axis=0)

        poses_o = np.stack([quat_to_mat(p) for p in poses_o], axis=1)
        self._logger.debug(f"Loaded poses_o: {poses_o.shape}")
        self._logger.debug(f"Loaded verts_m: {len(verts_m)} frames")
        self._logger.debug(f"Loaded joints_m: {len(joints_m)} frames")
        self._logger.debug(f"Loaded faces_m: {len(faces_m)} frames")
        
        # Create colors for hands (left hand: blue, right hand: red)
        colors_m = []
        for i in range(num_frames):
            verts_left, faces_left = left_hand_meshes[i]
            verts_right, faces_right = right_hand_meshes[i]
            
            if verts_left.shape[0] == 0 and verts_right.shape[0] == 0:
                # No hands available
                colors_m.append(np.zeros((0, 3), dtype=np.float32))
            elif verts_left.shape[0] == 0:
                # Only right hand
                right_colors = np.full((verts_right.shape[0], 3), [1, 0, 0], dtype=np.float32)
                colors_m.append(right_colors)
            elif verts_right.shape[0] == 0:
                # Only left hand
                left_colors = np.full((verts_left.shape[0], 3), [0, 0, 1], dtype=np.float32)
                colors_m.append(left_colors)
            else:
                # Both hands available
                left_colors = np.full((verts_left.shape[0], 3), [0, 0, 1], dtype=np.float32)
                right_colors = np.full((verts_right.shape[0], 3), [1, 0, 0], dtype=np.float32)
                colors_m.append(np.concatenate([left_colors, right_colors], axis=0))
        
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
        # self.render_optimized_poses()

        self._logger.info("=" * 100)
        self._logger.info(f"Joint Pose Solver Done!!! ({time.time() - t_s:.2f}s)")
        self._logger.info("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Pose Solver")
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run the solver in debug mode."
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder path.")

    solver = JointPoseSolver(args.sequence_folder, args.debug)
    solver.run()
