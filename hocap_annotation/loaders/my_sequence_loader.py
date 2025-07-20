import numpy as np
import h5py
import yaml
from pathlib import Path
import torch
from ..layers import MANOGroupLayer, ObjectGroupLayer
from ..utils import *

class MySequenceLoader:
    """
    Class for loading and processing sequence data from .h5 and h5 mask files (or .npy fallback), matching MyClusterLoader structure.
    """
    def __init__(
        self,
        sequence_folder: str,
        load_mano: bool = False,
        load_object: bool = False,
        in_world: bool = True,
        device: str = "cuda",
    ):
        self._data_folder = Path(sequence_folder)
        self._folder_name = self._data_folder.parent.name
        self._sequence_name = self._data_folder.name
        print(f"[INFO] folder_name: {self._folder_name}, sequence_name: {self._sequence_name}")
        self._seg_folder = self._data_folder.parent.parent / f"{self._folder_name}_annotated" / self._sequence_name / "tool_masks"
        self._object_masks_folder = self._data_folder.parent.parent / f"{self._folder_name}_annotated" / self._sequence_name / "object_masks"
        if not self._seg_folder.exists():
            self._seg_folder = self._data_folder.parent.parent / f"{self._folder_name}_annotated" / self._sequence_name / "masks"
        self._load_mano = load_mano
        self._load_object = load_object
        self._in_world = in_world
        self._device = device
        self._load_metadata()
        self._load_h5_and_masks()
        self._mano_group_layer = self._init_mano_group_layer()
        self._object_group_layer = self._init_object_group_layer()
        self._rays = self._create_3d_rays()
        self._M2world = torch.bmm(self._rs_Ks, self._extr2world_inv[:, :3, :])
        self._frame_id = -1
        self._points = torch.zeros((self.num_cameras, self._rs_height * self._rs_width, 3), dtype=torch.float32, device=self._device)
        self._colors = torch.zeros((self.num_cameras, self._rs_height * self._rs_width, 3), dtype=torch.float32, device=self._device)
        self._masks = torch.zeros((self.num_cameras, self._rs_height * self._rs_width), dtype=torch.bool, device=self._device)

    def _load_h5_and_masks(self):
        h5_files = list(self._data_folder.glob('*.h5'))
        assert len(h5_files) > 0, f"No .h5 file found in {self._data_folder}"
        h5_path = h5_files[0]
        with h5py.File(h5_path, 'r') as f:
            self._all_colors = f["imgs"][:]  # (N, num_cams, H, W, 3)
            self._all_depths = f["depths"][:] * 0.001  # (N, num_cams, H, W), convert to meters
        self._num_frames, self._num_cams = self._all_colors.shape[:2]
        self._rs_height, self._rs_width = self._all_colors.shape[2:4]
        self._all_masks = self._load_masks_from_folder(self._seg_folder, self._num_frames, self._num_cams, h5_name="masks.h5", h5_dataset="masks")
        if self._object_masks_folder.exists():
            self._all_object_masks = self._load_masks_from_folder(self._object_masks_folder, self._num_frames, self._num_cams, h5_name="object_masks.h5", h5_dataset="object_masks")
        else:
            self._all_object_masks = None

    def _load_masks_from_folder(self, mask_root_dir, num_frames, num_cams, h5_name=None, h5_dataset=None):
        mask_root_dir = Path(mask_root_dir)
        if h5_name is not None and (mask_root_dir / h5_name).exists():
            with h5py.File(mask_root_dir / h5_name, 'r') as f:
                masks = f[h5_dataset][:]
            print(f"[INFO] Loaded masks from {mask_root_dir / h5_name}")
            return masks
        all_masks = []
        for frame_idx in range(num_frames):
            frame_masks = []
            for cam_idx in range(num_cams):
                cam_folder = mask_root_dir / f"cam{cam_idx:02d}.mp4"
                npy_path = cam_folder / f"{frame_idx}.npy"
                if not npy_path.exists():
                    frame_masks.append(np.zeros((self._rs_height, self._rs_width), dtype=np.uint8))
                    continue
                mask = np.load(npy_path)
                frame_masks.append(mask)
            all_masks.append(frame_masks)
        all_masks = np.array(all_masks)  # (N, num_cams, H, W)
        return all_masks

    def _load_metadata(self):
        file_path = self._data_folder / "meta.yaml"
        data = read_data_from_yaml(file_path)
        self._models_folder = Path(data["models_folder"])
        self._calibration_yaml_path = Path(data["calibration_yaml_path"])
        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]
        self._subject_id = data["subject_id"]
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cams = len(self._rs_serials)
        self.have_hl = data["have_hololens"]
        self.have_mano = data["have_mano"]
        if self.have_hl:
            self._hl_serial = data["hololens"]["serial"]
            self._hl_pv_width = data["hololens"]["pv_width"]
            self._hl_pv_height = data["hololens"]["pv_height"]
        self._object_textured_files = [self._models_folder / obj_id / "textured_mesh.obj" for obj_id in self._object_ids]
        self._object_cleaned_files = [self._models_folder / obj_id / "cleaned_mesh_10000.obj" for obj_id in self._object_ids]
        self._thresholds = data.get("thresholds", [-0.3, 0.3, -0.3, 0.3, 0.5, 0.95])
        if self.have_mano:
            self._mano_beta = torch.tensor(data["betas"], dtype=torch.float32, device=self._device)
        self._load_camera_params_from_yaml()
        self._crop_lim = data.get("thresholds", [-0.3, 0.3, -0.3, 0.3, -0.2, 0.4])


    def _load_camera_params_from_yaml(self):
        with open(self._calibration_yaml_path, 'r') as f:
            data = yaml.load(f)
        self._rs_serials = [str(cam['camera_id']).zfill(2) for cam in data]
        rs_Ks = np.stack([np.array(cam['color_intrinsic_matrix'], dtype=np.float32) for cam in data], axis=0)
        rs_Ks_inv = np.stack([np.linalg.inv(K) for K in rs_Ks], axis=0)
        self._rs_Ks = torch.from_numpy(rs_Ks).to(self._device)
        self._rs_Ks_inv = torch.from_numpy(rs_Ks_inv).to(self._device)
        extr2world = np.stack([np.array(cam['transformation'], dtype=np.float32) for cam in data], axis=0)
        extr2world_inv = np.stack([np.linalg.inv(np.array(cam['transformation'], dtype=np.float32)) for cam in data], axis=0)
        self._extr2world = torch.from_numpy(extr2world).to(self._device)
        self._extr2world_inv = torch.from_numpy(extr2world_inv).to(self._device)

    def _create_3d_rays(self) -> torch.Tensor:
        """Creates 3D rays for deprojecting depth images to 3D space."""

        def create_2d_coords() -> torch.Tensor:
            xv, yv = torch.meshgrid(
                torch.arange(self._rs_width),
                torch.arange(self._rs_height),
                indexing="xy",
            )
            coord_2d = torch.stack(
                (xv, yv, torch.ones_like(xv)), dim=0
            ).float()  # (3, H, W)
            coords_2d = (
                coord_2d.unsqueeze(0)
                .repeat(self._num_cams, 1, 1, 1)
                .view(self._num_cams, 3, -1)
            )  # (N, 3, H*W)
            coords_2d = coords_2d.to(self._device)
            return coords_2d

        coords_2d = create_2d_coords()
        return torch.bmm(self._rs_Ks_inv, coords_2d)  # (N, 3, H*W)

    def _init_mano_group_layer(self):
        """Initialize the MANO group layer."""
        if not self._load_mano:
            return None

        mano_group_layer = MANOGroupLayer(
            self._mano_sides, [self._mano_beta.cpu().numpy()] * len(self._mano_sides)
        ).to(self._device)
        return mano_group_layer

    def _init_object_group_layer(self):
        """Initialize the object group layer."""
        if not self._load_object:
            print("[DEBUG] Object group layer not loaded.")
            return None

        verts, faces, norms = [], [], []
        for obj_file in self.object_cleaned_mesh_files:
            m = trimesh.load(obj_file, process=False)
            verts.append(m.vertices)
            faces.append(m.faces)
            norms.append(m.vertex_normals)
        print(f"[DEBUG] Loaded {len(verts)} object meshes with {sum(len(v) for v in verts)} vertices.")
        print(f"[DEBUG] Loaded {len(faces)} object meshes with {sum(len(f) for f in faces)} faces.")
        print(f"[DEBUG] Loaded {len(norms)} object meshes with {sum(len(n) for n in norms)} normals.")
        object_group_layer = ObjectGroupLayer(verts, faces, norms).to(self._device)
        return object_group_layer

    def _deproject(self, colors, depths) -> tuple:
        """
        Deprojects depth images to 3D points.

        Args:
            colors (np.ndarray): List of color images, [N, H, W, 3], dtype=float32.
            depths (np.ndarray): List of depth images, [N, H, W], dtype=np.float32.

        Returns:
            tuple: Colors, 3D points, and masks.
        """
        # Process color images
        colors = torch.from_numpy(colors.reshape(self._num_cams, -1, 3)).to(
            self._device
        ).float()  # [N, H*W, 3]

        # Process depth images
        depths = torch.from_numpy(depths.reshape(self._num_cams, 1, -1)).to(
            self._device
        ).float()  # [N, 1, H*W]

        # Deproject depth images to 3D points in camera frame
        pts_c = self._rays * depths  # [N, 3, H*W]
        # Transform 3D points from camera frame to world frame
        pts = torch.baddbmm(
            self._extr2world[:, :3, 3].unsqueeze(2),
            self._extr2world[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # (N, H*W, 3)

        # Crop 3D points
        mx1 = pts[..., 0] > self._thresholds[0]
        mx2 = pts[..., 0] < self._thresholds[1]
        my1 = pts[..., 1] > self._thresholds[2]
        my2 = pts[..., 1] < self._thresholds[3]
        mz1 = pts[..., 2] > self._thresholds[4]
        mz2 = pts[..., 2] < self._thresholds[5]
        masks = mx1 & mx2 & my1 & my2 & mz1 & mz2

        # Transform 3D points from world frame to master frame if necessary
        if not self._in_world:
            pts = torch.baddbmm(
                self._extr2world[:, :3, 3].unsqueeze(2),
                self._extr2world[:, :3, :3],
                pts_c,
            ).permute(
                0, 2, 1
            )  # [N, H*W, 3]

        return colors, pts, masks

    def _update_pcd(self, frame_id: int):
        """Update point cloud data."""
        colors, points, masks = self._deproject(
            self._all_colors[frame_id],
            self._all_depths[frame_id],
        )
        self._points.copy_(points)
        self._colors.copy_(colors)
        self._masks.copy_(masks)

    def get_rgb_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]."""
        # This method is no longer needed as data is in memory
        return self._all_colors[frame_id, self._rs_serials.index(serial)]

    def get_depth_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get depth image in numpy format, dtype=uint16, [H, W]."""
        # This method is no longer needed as data is in memory
        return self._all_depths[frame_id, self._rs_serials.index(serial)]

    def get_mask_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get mask image in numpy format, dtype=uint8, [H, W]."""
        # This method is no longer needed as data is in memory
        return self._all_masks[frame_id, self._rs_serials.index(serial)]

    def object_group_layer_forward(
        self, poses, subset=None
    ) -> tuple:
        """Forward pass for the object group layer."""
        if self._object_group_layer is None:
            raise RuntimeError("Object group layer is not loaded. Set load_object=True.")
        p = torch.cat(poses, dim=1)
        v, n = self._object_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

    def mano_group_layer_forward(
        self, poses, subset=None
    ) -> tuple:
        """Forward pass for the MANO group layer."""
        if self._mano_group_layer is None:
            raise RuntimeError("MANO group layer is not loaded. Set load_mano=True.")
        p = torch.cat(poses, dim=1)
        v, j = self._mano_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            j = j.squeeze(0)
        return v, j

    def step(self):
        """Step to the next frame."""
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id: int):
        """Step to a specific frame."""
        self._frame_id = frame_id % self._num_frames
        self._update_pcd(self._frame_id)

    # Property methods for access to class attributes

    @property
    def sequence_folder(self) -> str:
        return str(self._data_folder)

    @property
    def load_mano(self) -> bool:
        return self._load_mano

    @property
    def load_object(self) -> bool:
        return self._load_object

    @property
    def in_world(self) -> bool:
        return self._in_world

    @property
    def device(self) -> str:
        return self._device

    @property
    def object_ids(self) -> list:
        return self._object_ids

    @property
    def group_id(self) -> str:
        return self._object_ids[0].split("_")[0]

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def rs_width(self) -> int:
        return self._rs_width

    @property
    def rs_height(self) -> int:
        return self._rs_height

    @property
    def rs_serials(self) -> list:
        return self._rs_serials

    @property
    def num_cameras(self) -> int:
        return self._num_cams

    @property
    def holo_serial(self) -> list:
        return self._hl_serial

    @property
    def holo_pv_width(self) -> int:
        return self._hl_pv_width

    @property
    def holo_pv_height(self) -> int:
        return self._hl_pv_height

    @property
    def mano_beta(self) -> torch.Tensor:
        return self._mano_beta

    @property
    def mano_sides(self) -> list:
        return self._mano_sides

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._rs_Ks

    @property
    def intrinsics_inv(self) -> torch.Tensor:
        return self._rs_Ks_inv

    @property
    def extrinsics2world(self) -> torch.Tensor:
        return self._extr2world

    @property
    def extrinsics2world_inv(self) -> torch.Tensor:
        return self._extr2world_inv

    @property
    def M2world(self) -> torch.Tensor:
        """camera to world transformation matrix"""
        return self._M2world

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def mano_group_layer(self):
        return self._mano_group_layer

    @property
    def object_group_layer(self):
        return self._object_group_layer

    @property
    def object_textured_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/textured_mesh.obj")
            for object_id in self._object_ids
        ]

    @property
    def object_cleaned_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/cleaned_mesh_10000.obj")
            for object_id in self._object_ids
        ]

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @property
    def colors(self) -> torch.Tensor:
        return self._colors

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def points_map(self) -> torch.Tensor:
        return self._points.view(self._num_cams, self._rs_height, self._rs_width, 3)

    @property
    def colors_map(self) -> torch.Tensor:
        return self._colors.view(self._num_cams, self._rs_height, self._rs_width, 3)

    @property
    def masks_map(self) -> torch.Tensor:
        return self._masks.view(self._num_cams, self._rs_height, self._rs_width)
