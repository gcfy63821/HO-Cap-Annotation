from ..utils import *
import numpy as np
import h5py
from pathlib import Path
import yaml
import open3d as o3d

class MyClusterLoader:
    """
    Data loader for HO-Cap dataset. Loads images, depths, masks, and camera parameters directly from the original data and calibration YAML.
    """
    def __init__(self, sequence_folder) -> None:
        """
        Initialize the loader.
        Args:
            sequence_folder (str): Path to the folder containing the .h5 file and mask folders.
        """
        self._data_folder = Path(sequence_folder)
        # Updated path parsing for new structure: videos_0901/taskname/xxxvideoname
        self._sequence_name = self._data_folder.name  # xxxvideoname
        task_folder = self._data_folder.parent  # .../videos_0901/taskname
        self._task_name = task_folder.name  # taskname
        video_root_folder = task_folder.parent  # .../videos_0901
        self._folder_name = video_root_folder.name  # videos_0901
        
        print(f"[INFO] folder_name: {self._folder_name}, task_name: {self._task_name}, sequence_name: {self._sequence_name}")
        
        # Create annotated paths with taskname included: videos_0901_annotated/taskname/xxxvideoname
        annotated_base_folder = f"{self._folder_name}_annotated"  # videos_0901_annotated
        annotated_root = video_root_folder.parent / annotated_base_folder  # .../videos_0901_annotated
        annotated_sequence_path = annotated_root / self._task_name / self._sequence_name  # .../videos_0901_annotated/taskname/xxxvideoname
        self._annotated_folder = annotated_sequence_path
        self._annotated_root = annotated_root
        tool_masks_folder = annotated_sequence_path / "tool_masks"
        masks_folder = annotated_sequence_path / "masks"
        self._object_masks_folder = annotated_sequence_path / "object_masks"
        print(f"[INFO] tool mask folder_name: {tool_masks_folder}, is dir: {tool_masks_folder.is_dir()}")
        
        if tool_masks_folder.exists() and tool_masks_folder.is_dir():
            self._seg_folder = tool_masks_folder
        elif masks_folder.exists() and masks_folder.is_dir():
            self._seg_folder = masks_folder
        else:
            raise FileNotFoundError(f"No tool_masks or object_masks folder found in {self._folder_name}_annotated/{self._task_name}/{self._sequence_name}")
        
        self._load_metadata()
        self._load_h5_and_masks()

    def _load_h5_and_masks(self):
        """
        Load all color images, depth images, and masks into memory from .h5 and h5 mask files (or .npy files if h5 not present).
        Sets:
            self._all_colors: np.ndarray, shape (N, num_cams, H, W, 3)
            self._all_depths: np.ndarray, shape (N, num_cams, H, W)
            self._all_masks: np.ndarray, shape (N, num_cams, H, W)
            self._all_object_masks: np.ndarray or None, shape (N, num_cams, H, W)
        """
        h5_files = list(self._data_folder.glob('*.h5'))
        assert len(h5_files) > 0, f"No .h5 file found in {self._data_folder}"
        h5_path = h5_files[0]
        with h5py.File(h5_path, 'r') as f:
            # Select only active cameras from h5 using calibration indices
            self._all_colors = f["imgs"][:, self._cam_h5_indices]  # (N, active_cams, H, W, 3)
            self._all_depths = f["depths"][:, self._cam_h5_indices]  # (N, active_cams, H, W)
            self._all_depths = self._all_depths * 0.001
        self._num_frames, self._num_cams = self._all_colors.shape[:2]
        self._rs_height, self._rs_width = self._all_colors.shape[2:4]
        print(f"[INFO] Loaded h5: {self._num_frames} frames, {self._num_cams} active cameras (from h5 indices {self._cam_h5_indices})")
        self._all_masks = self._load_masks_from_folder(self._seg_folder, self._num_frames, self._num_cams, h5_name="masks.h5", h5_dataset="masks")
        object_mask_dir = self._object_masks_folder
        if object_mask_dir.exists():
            self._all_object_masks = self._load_masks_from_folder(object_mask_dir, self._num_frames, self._num_cams, h5_name="object_masks.h5", h5_dataset="object_masks")
        else:
            self._all_object_masks = None

    def _load_masks_from_folder(self, mask_root_dir, num_frames, num_cams, h5_name=None, h5_dataset=None):
        """
        Load all masks from an h5 file if present, otherwise from a folder of .npy files.
        Args:
            mask_root_dir (Path): Path to the root mask folder.
            num_frames (int): Number of frames.
            num_cams (int): Number of cameras.
            h5_name (str or None): Name of the h5 file to look for.
            h5_dataset (str or None): Name of the dataset in the h5 file.
        Returns:
            np.ndarray: Array of shape (num_frames, num_cams, H, W)
        """
        mask_root_dir = Path(mask_root_dir)
        if h5_name is not None and (mask_root_dir / h5_name).exists():
            with h5py.File(mask_root_dir / h5_name, 'r') as f:
                masks = f[h5_dataset][:, self._cam_h5_indices]  # select active cameras
            print(f"[INFO] Loaded masks from {mask_root_dir / h5_name}, selected h5 indices {self._cam_h5_indices}, shape: {masks.shape}")
            return masks
        # fallback to npy loading
        # mask npy files are indexed by original video frame number
        # npy folder names use original calibration index (e.g. cam2_rgb), not active camera index
        all_masks = []
        for frame_idx in range(num_frames):
            original_frame_idx = frame_idx + self._start_frame
            frame_masks = []
            for cam_idx in range(num_cams):
                h5_idx = self._cam_h5_indices[cam_idx]
                cam_folder = mask_root_dir / f"cam{h5_idx}_rgb"
                # Try different filename formats using original frame index
                npy_path = None
                for fmt in [f"{original_frame_idx:04d}.npy", f"{original_frame_idx}.npy"]:
                    test_path = cam_folder / fmt
                    if test_path.exists():
                        npy_path = test_path
                        break
                
                if npy_path is None or not npy_path.exists():
                    frame_masks.append(np.zeros((self._rs_height, self._rs_width), dtype=np.uint8))
                    continue
                
                mask = np.load(npy_path)
                # Handle different mask shapes: (1, H, W) -> (H, W), or (H, W) -> (H, W)
                if mask.ndim == 3:
                    mask = mask.squeeze(0)  # Remove first dimension if present
                elif mask.ndim != 2:
                    print(f"[WARNING] Unexpected mask shape {mask.shape} from {npy_path}, using zeros")
                    mask = np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)
                
                # Resize mask if dimensions don't match
                if mask.shape[0] != self._rs_height or mask.shape[1] != self._rs_width:
                    try:
                        import cv2
                        mask = cv2.resize(mask, (self._rs_width, self._rs_height), interpolation=cv2.INTER_NEAREST)
                    except ImportError:
                        from scipy.ndimage import zoom
                        scale_h = self._rs_height / mask.shape[0]
                        scale_w = self._rs_width / mask.shape[1]
                        mask = zoom(mask, (scale_h, scale_w), order=0).astype(mask.dtype)
                
                frame_masks.append(mask)
            all_masks.append(frame_masks)
        all_masks = np.array(all_masks)  # (N, num_cams, H, W)
        return all_masks

    def _depth2xyz(self, depth, K, T=None):
        """
        Convert a depth image to a 3D point cloud in camera/world coordinates.
        Args:
            depth (np.ndarray): Depth image, shape (H, W)
            K (np.ndarray): Camera intrinsic matrix, shape (3, 3)
            T (np.ndarray or None): Extrinsic matrix, shape (4, 4) or (3, 4). If None, returns camera coordinates.
        Returns:
            np.ndarray: Point cloud, shape (H*W, 3)
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth.flatten()
        depth_flat[depth_flat < 0] = 0
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_norm = (u_flat - cx) / fx
        y_norm = (v_flat - cy) / fy
        x = depth_flat * x_norm
        y = depth_flat * y_norm
        z = depth_flat
        xyz = np.stack((x, y, z), axis=1)  # camera space
        if T is not None:
            xyz = xyz @ T[:3, :3].T + T[:3, 3]
        return xyz

    def get_init_translation(self, frame_id, serials, object_idx, kernel_size=3):
        """
        Estimate the initial translation (center) of the object in each camera's coordinate system for a given frame.
        Args:
            frame_id (int): Frame index.
            serials (list[str]): List of camera serials to use.
            object_idx (int): Object index (0-based).
            kernel_size (int): Mask erosion kernel size.
        Returns:
            tuple: (centers, pcd)
                centers: np.ndarray, shape (len(serials), 3), object center in each camera's coordinates
                pcd: open3d.geometry.PointCloud, filtered point cloud
        """
        masks = [
            self.get_mask(s, frame_id, object_idx, kernel_size)
            for s in self._rs_serials
        ]
        depths = [self.get_depth(s, frame_id) for s in self._rs_serials]
        pts = [
            self._depth2xyz(depth, K, T)
            for depth, K, T in zip(depths, self._rs_Ks, self._extr2world)
        ]
        pts = [pt[mask.flatten().astype(bool)] for pt, mask in zip(pts, masks)]
        pts = np.concatenate(pts, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) < 100:
            print(f"[WARNING] Not enough points for frame {frame_id}, serials {serials} when get_init_translation")
            return [None] * len(serials), pcd
        center = pcd.get_center()
        # transform to each camera coordinate system
        centers = []
        for serial in serials:
            extr = self._extr2world_inv[self._rs_serials.index(serial)]
            center_cam = center @ extr[:3, :3].T + extr[:3, 3]
            centers.append(center_cam)
        centers = np.stack(centers, axis=0, dtype=np.float32)
        return centers, pcd

    def _load_metadata(self):
        """
        Load meta.yaml and extract dataset and calibration information.
        Sets:
            self._models_folder, self._calibration_yaml_path, self._num_frames, self._object_ids, etc.
        """
        file_path = self._data_folder / "meta.yaml"
        data = read_data_from_yaml(file_path)
        self._models_folder = Path(data["models_folder"])
        self._calibration_yaml_path = Path(data["calibration_yaml_path"])
        self._load_camera_params_from_yaml()
        # _rs_serials now has ALL cameras from calibration YAML (e.g. ['00'..'07'])
        # _rs_Ks, _extr2world, _extr2world_inv indexed by calibration order
        all_calib_serials = list(self._rs_serials)  # save full list

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._subject_id = data["subject_id"]
        meta_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]

        # Compute mapping: meta serial -> h5/calibration index
        # e.g. meta_serials=['02','03',...'07'] -> _cam_h5_indices=[2,3,...,7]
        self._cam_h5_indices = [all_calib_serials.index(s) for s in meta_serials]
        print(f"[INFO] Active cameras: {meta_serials}, h5 indices: {self._cam_h5_indices}")

        # Filter Ks and extrinsics to only active cameras
        self._rs_Ks = self._rs_Ks[self._cam_h5_indices]
        self._extr2world = self._extr2world[self._cam_h5_indices]
        self._extr2world_inv = self._extr2world_inv[self._cam_h5_indices]
        self._rs_serials = meta_serials
        self.have_hl = data["have_hololens"]
        self.have_mano = data["have_mano"]
        if self.have_hl:
            self._hl_serial = data["hololens"]["serial"]
            self._hl_height = data["hololens"]["pv_height"]
            self._hl_width = data["hololens"]["pv_width"]
        self._object_textured_files = [
            self._models_folder / obj_id / "textured_mesh.obj"
            for obj_id in self._object_ids
        ]
        self._object_cleaned_files = [
            self._models_folder / obj_id / "cleaned_mesh_10000.obj"
            for obj_id in self._object_ids
        ]
        self._thresholds = data.get("thresholds", [-0.3, 0.3, -0.3, 0.3, -0.2, 0.4])
        self._start_frame = data.get("start_frame", 0)
        print(f"[DEBUG] thresholds: {self._thresholds}, start_frame: {self._start_frame}")
        if self.have_mano:
            self._mano_beta = np.array(data["betas"], dtype=np.float32)

    def _load_camera_params_from_yaml(self):
        """
        Load camera intrinsics and extrinsics from the original calibration YAML file.
        Sets:
            self._rs_serials, self._rs_Ks, self._extr2world, self._extr2world_inv
        """
        with open(self._calibration_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self._rs_serials = [str(cam['camera_id']).zfill(2) for cam in data]
        self._rs_Ks = np.stack([np.array(cam['color_intrinsic_matrix'], dtype=np.float32) for cam in data], axis=0)
        self._extr2world = np.stack([np.array(cam['transformation'], dtype=np.float32) for cam in data], axis=0)
        self._extr2world_inv = np.stack([np.linalg.inv(np.array(cam['transformation'], dtype=np.float32)) for cam in data], axis=0)

    def get_color(self, serial, frame_id):
        """
        Get RGB image for a given camera and frame.
        Args:
            serial (str): Camera serial number (e.g., '00').
            frame_id (int): Frame index.
        Returns:
            np.ndarray: RGB image, shape (H, W, 3), dtype=uint8
        """
        cam_idx = self._rs_serials.index(serial)
        return self._all_colors[frame_id, cam_idx]

    def get_depth(self, serial, frame_id):
        """
        Get depth image for a given camera and frame.
        Args:
            serial (str): Camera serial number (e.g., '00').
            frame_id (int): Frame index.
        Returns:
            np.ndarray: Depth image, shape (H, W), dtype=float32
        """
        cam_idx = self._rs_serials.index(serial)
        return self._all_depths[frame_id, cam_idx]

    def get_mask(self, serial, frame_id, object_idx, kernel_size=1):
        """
        Get binary mask for a given camera, frame, and object.
        Args:
            serial (str): Camera serial number (e.g., '00').
            frame_id (int): Frame index.
            object_idx (int): Object index (0-based).
            kernel_size (int): Erosion kernel size (default 1, no erosion).
        Returns:
            np.ndarray: Binary mask, shape (H, W), dtype=uint8
        """
        cam_idx = self._rs_serials.index(serial)
        mask = self._all_masks[frame_id, cam_idx]
        mask = (mask == (object_idx + 1)).astype(np.uint8)
        # if kernel_size > 1:
        #     mask = erode_mask(mask, kernel_size)
        # mask shape: (1, 480, 640) --> (480, 640)
        if mask.ndim == 3:
            mask = mask[0]
        return mask

    def get_object_mask(self, serial, frame_id):
        """
        Get object mask for a given camera and frame.
        Args:
            serial (str): Camera serial number (e.g., '00').
            frame_id (int): Frame index.
        Returns:
            np.ndarray: Object mask, shape (H, W), dtype=uint8
        """
        cam_idx = self._rs_serials.index(serial)
        if self._all_object_masks is not None:
            return self._all_object_masks[frame_id, cam_idx]
        else:
            return np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)

    def get_all_colors(self):
        """
        Get all color images.
        Returns:
            np.ndarray: All color images, shape (num_frames, num_cams, H, W, 3)
        """
        return self._all_colors

    def get_all_depths(self):
        """
        Get all depth images.
        Returns:
            np.ndarray: All depth images, shape (num_frames, num_cams, H, W)
        """
        return self._all_depths

    def get_all_masks(self, object_idx=None, kernel_size=1):
        """
        Get all masks, optionally for a specific object and with erosion.
        Args:
            object_idx (int or None): Object index (0-based) or None for raw masks.
            kernel_size (int): Erosion kernel size (default 1, no erosion).
        Returns:
            np.ndarray: Masks, shape (num_frames, num_cams, H, W)
        """
        if object_idx is None and kernel_size == 1:
            return self._all_masks
        num_frames, num_cams, H, W = self._all_masks.shape
        masks = np.zeros_like(self._all_masks, dtype=np.uint8)
        for cam_idx in range(num_cams):
            for frame_id in range(num_frames):
                mask = self._all_masks[frame_id, cam_idx]
                if object_idx is not None:
                    mask = (mask == (object_idx + 1)).astype(np.uint8)
                if kernel_size > 1:
                    mask = erode_mask(mask, kernel_size)
                masks[frame_id, cam_idx] = mask
        return masks

    def get_all_object_masks(self, kernel_size=1):
        """
        Get all object masks, optionally with erosion.
        Args:
            kernel_size (int): Erosion kernel size (default 1, no erosion).
        Returns:
            np.ndarray: Object masks, shape (num_frames, num_cams, H, W)
        """
        if self._all_object_masks is None:
            return np.zeros_like(self._all_masks, dtype=np.uint8)
        num_frames, num_cams, H, W = self._all_object_masks.shape
        masks = np.zeros_like(self._all_object_masks, dtype=np.uint8)
        for cam_idx in range(num_cams):
            for frame_id in range(num_frames):
                mask = self._all_object_masks[frame_id, cam_idx]
                if kernel_size > 1:
                    mask = erode_mask(mask, kernel_size)
                masks[frame_id, cam_idx] = mask
        return masks

    def get_valid_seg_serials(self):
        """
        Get list of camera serials for which segmentation masks exist.
        Checks for the first frame's mask file using start_frame offset.
        Returns:
            list[str]: List of valid camera serials.
        """
        valid_serials = []
        for cam_idx, serial in enumerate(self._rs_serials):
            h5_idx = self._cam_h5_indices[cam_idx]
            cam_folder = self._seg_folder / f"cam{h5_idx}_rgb"
            # Check using original frame index (with start_frame offset)
            found = False
            for fmt in [f"{self._start_frame:04d}.npy", f"{self._start_frame}.npy"]:
                if (cam_folder / fmt).exists():
                    found = True
                    break
            if found:
                valid_serials.append(serial)
        return valid_serials

    def get_seg_color_index_map(self):
        """
        Get mapping from segmentation color to index.
        Returns:
            dict: Mapping from color.rgb to index.
        """
        color_index_map = {color.rgb: idx for idx, color in enumerate(HO_CAP_SEG_COLOR)}
        return color_index_map

    def get_object_seg_color(self, object_id):
        """
        Get segmentation color for a given object id.
        Args:
            object_id (str): Object id.
        Returns:
            tuple: RGB color.
        """
        idx = self._object_ids.index(object_id) + 1
        color = HO_CAP_SEG_COLOR[idx].rgb
        return color

    def get_mano_seg_color(self, side):
        """
        Get segmentation color for a given hand side ('left' or 'right').
        Args:
            side (str): 'left' or 'right'.
        Returns:
            tuple: RGB color.
        """
        idx = 5 if side == "right" else 6
        color = HAND_COLORS[idx].rgb
        return color

    # Properties for dataset attributes
    @property
    def cam_h5_indices(self):
        """Mapping from active camera index to h5 camera dimension index."""
        return self._cam_h5_indices

    @property
    def num_frames(self):
        """Number of frames in the sequence."""
        return self._num_frames

    @property
    def rs_serials(self):
        """List of camera serials."""
        return self._rs_serials

    @property
    def rs_width(self):
        """Width of the realsense images."""
        return self._rs_width

    @property
    def rs_height(self):
        """Height of the realsense images."""
        return self._rs_height

    @property
    def hl_serial(self):
        """Hololens serial (if present)."""
        return self._hl_serial

    @property
    def hl_width(self):
        """Hololens image width (if present)."""
        return self._hl_width

    @property
    def hl_height(self):
        """Hololens image height (if present)."""
        return self._hl_height

    @property
    def rs_Ks(self):
        """Camera intrinsics for all realsense cameras."""
        return self._rs_Ks

    @property
    def extr2world(self):
        """Extrinsic matrices for all realsense cameras."""
        return self._extr2world

    @property
    def extr2world_inv(self):
        """Inverse extrinsic matrices for all realsense cameras."""
        return self._extr2world_inv

    @property
    def mano_sides(self):
        """List of hand sides (e.g., ['left', 'right'])."""
        return self._mano_sides

    @property
    def mano_beta(self):
        """MANO shape parameters (if present)."""
        return self._mano_beta

    @property
    def subject_id(self):
        """Subject id for the sequence."""
        return self._subject_id

    @property
    def object_ids(self):
        """List of object ids in the sequence."""
        return self._object_ids

    @property
    def object_textured_files(self):
        """List of textured mesh file paths for objects."""
        return self._object_textured_files

    @property
    def object_cleaned_files(self):
        """List of cleaned mesh file paths for objects."""
        return self._object_cleaned_files

    @property
    def x_threshold(self):
        """X-axis threshold for filtering points."""
        return self._thresholds[0:2]

    @property
    def y_threshold(self):
        """Y-axis threshold for filtering points."""
        return self._thresholds[2:4]

    @property
    def z_threshold(self):
        """Z-axis threshold for filtering points."""
        return self._thresholds[4:6]

    @property
    def task_name(self):
        """Task name for the sequence."""
        return self._task_name
