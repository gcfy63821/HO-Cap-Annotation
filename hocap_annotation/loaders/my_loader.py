from ..utils import *
import numpy as np


class MyLoader:

    def __init__(self, sequence_folder) -> None:
        self._data_folder = Path(sequence_folder)
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._models_folder = self._data_folder.parent.parent / "models"
        self._seg_folder = self._data_folder / "processed/segmentation/sam2"

        self._load_metadata()
        self._load_all_images_to_memory()

    def _load_all_images_to_memory(self):
        """一次性加载所有color、depth、mask到内存"""
        num_frames = self._num_frames
        num_cams = len(self._rs_serials)
        H, W = self._rs_height, self._rs_width

        # 颜色图
        self._all_colors = np.zeros((num_frames, num_cams, H, W, 3), dtype=np.uint8)
        # 深度图
        self._all_depths = np.zeros((num_frames, num_cams, H, W), dtype=np.float32)
        # mask
        self._all_masks = np.zeros((num_frames, num_cams, H, W), dtype=np.uint8)
        # object mask
        self._all_object_masks = np.zeros((num_frames, num_cams, H, W), dtype=np.uint8) if self.have_mano else None

        for cam_idx, serial in enumerate(self._rs_serials):
            for frame_id in range(num_frames):
                # color
                color_path = self._data_folder / serial / f"color_{frame_id:06d}.jpg"
                self._all_colors[frame_id, cam_idx] = read_rgb_image(color_path)
                # depth
                depth_path = self._data_folder / serial / f"depth_{frame_id:06d}.png"
                self._all_depths[frame_id, cam_idx] = read_depth_image(depth_path, 1000.0)
                # mask
                mask_path = self._seg_folder / serial / "mask" / f"mask_{frame_id:06d}.png"
                if mask_path.exists():
                    mask_img = read_mask_image(mask_path)
                    self._all_masks[frame_id, cam_idx] = mask_img
                else:
                    self._all_masks[frame_id, cam_idx] = np.zeros((H, W), dtype=np.uint8)
                # object mask
                object_mask_path = self._seg_folder / serial / "object_mask" / f"object_mask_{frame_id:06d}.png"
                if object_mask_path.exists():
                    object_mask_img = read_mask_image(object_mask_path)
                    self._all_object_masks = np.zeros((num_frames, num_cams, H, W), dtype=np.uint8)
                    self._all_object_masks[frame_id, cam_idx] = object_mask_img

    def _depth2xyz(self, depth, K, T=None):
        """Convert depth image to xyz point cloud"""
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
        # 保存原始点云 DEBUG
        # output_dir = "debug_outputs/"
        # raw_pcd_path = os.path.join(output_dir, f"frame_{frame_id}_raw.ply")
        # o3d.io.write_point_cloud(raw_pcd_path, pcd)
        # print(f"[INFO] Saved raw point cloud to {raw_pcd_path}")

        # Remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        # ####### DEBUG
        # filtered_pcd_path = os.path.join(output_dir, f"frame_{frame_id}_filtered.ply")
        # o3d.io.write_point_cloud(filtered_pcd_path, pcd)
        # print(f"[INFO] Saved filtered point cloud to {filtered_pcd_path}")

        # 可视化mask和depth（取第一个摄像头为例）
        # mask_img = (masks[0] * 255).astype(np.uint8)
        # mask_path = os.path.join(output_dir, f"frame_{frame_id}_mask_cam0.png")
        # cv2.imwrite(mask_path, mask_img)
        # print(f"[INFO] Saved mask image to {mask_path}")

        depth = depths[0]
        # depth 归一化到0-255，方便查看（注意不要用于计算）
        # depth_vis = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8) * 255
        # depth_vis = depth_vis.astype(np.uint8)
        # depth_path = os.path.join(output_dir, f"frame_{frame_id}_depth_cam0.png")
        # cv2.imwrite(depth_path, depth_vis)
        # print(f"[INFO] Saved depth visualization to {depth_path}")

        pts = np.asarray(pcd.points, dtype=np.float32)

        if len(pts) < 100:
            print(f"[WARNING] Not enough points for frame {frame_id}, serials {serials} when get_init_translation")
            return [None] * len(serials), pcd

        center = np.mean(pts, axis=0)

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
        file_path = self._data_folder / "meta.yaml"
        data = read_data_from_yaml(file_path)

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._subject_id = data["subject_id"]
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
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

        # Load thresholds from meta.yaml if present
        # X_THRESHOLD = (-0.3, 0.2)
        # Y_THRESHOLD = (-0.3, 0.3)
        # Z_THRESHOLD = (0.5, 0.95)
        # thresholds = data.get("thresholds", {})
        # self._thresholds = {
        #     "x": thresholds.get("x", [-0.3, 0.3]),
        #     "y": thresholds.get("y", [-0.3, 0.3]),
        #     "z": thresholds.get("z", [0.5, 0.95]),
        # }
        self._thresholds = data.get("thresholds", [-0.3, 0.3, -0.3, 0.3, 0.5, 0.95])

        # self._texture_files = [
        #     self._models_folder / obj_id / "textured_mesh_0.jpg"
        #     for obj_id in self._object_ids
        # ]

        # Load camera intrinsics
        self._load_intrinsics()

        # Load rs camera extrinsics
        self._load_extrinsics(data["extrinsics"])

        # Load MANO shape parameters
        if self.have_mano:
            self._load_mano_beta()

    def _load_intrinsics(self):
        def read_K_from_yaml(serial, cam_type="color"):
            file_path = self._calib_folder / "intrinsics" / f"{serial}.yaml"
            data = read_data_from_yaml(file_path)[cam_type]
            K = np.array(
                [
                    [data["fx"], 0.0, data["ppx"]],
                    [0.0, data["fy"], data["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        self._rs_Ks = np.stack(
            [read_K_from_yaml(serial) for serial in self._rs_serials],
            axis=0,
        )
        if self.have_hl:
            self._hl_K = read_K_from_yaml(self._hl_serial)

    def _load_extrinsics(self, file_name):
        def create_mat(values):
            return np.array(
                [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32
            )

        file_path = self._calib_folder / "extrinsics" / f"{file_name}"
        data = read_data_from_yaml(file_path)

        extrinsics = data["extrinsics"]
        extr2world = [create_mat(extrinsics[s]) for s in self._rs_serials]
        extr2world_inv = [np.linalg.inv(tt) for tt in extr2world]
        # let me test

        self._extr2world = np.stack(extr2world, axis=0)
        self._extr2world_inv = np.stack(extr2world_inv, axis=0)
        # self._extr2world = np.stack(extr2world_inv, axis=0)
        # self._extr2world_inv = np.stack(extr2world, axis=0)

    def _load_mano_beta(self):
        file_path = self._calib_folder / "mano" / f"{self._subject_id}.yaml"
        data = read_data_from_yaml(file_path)
        self._mano_beta = np.array(data["betas"], dtype=np.float32)

    def get_color(self, serial, frame_id):
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]，从内存读取"""
        cam_idx = self._rs_serials.index(serial)
        return self._all_colors[frame_id, cam_idx]

    def get_depth(self, serial, frame_id):
        """Get depth image in numpy format, dtype=float32, [H, W]，从内存读取"""
        cam_idx = self._rs_serials.index(serial)
        return self._all_depths[frame_id, cam_idx]

    def get_mask(self, serial, frame_id, object_idx, kernel_size=1):
        """Get mask image in numpy format, dtype=uint8, [H, W]，从内存读取并处理object_idx和腐蚀"""
        cam_idx = self._rs_serials.index(serial)
        mask = self._all_masks[frame_id, cam_idx]
        mask = (mask == (object_idx + 1)).astype(np.uint8)
        if kernel_size > 1:
            mask = erode_mask(mask, kernel_size)
        return mask

    def get_object_mask(self, serial, frame_id):
        """Get object mask image in numpy format, dtype=uint8, [H, W]，从内存读取"""
        cam_idx = self._rs_serials.index(serial)
        if self._all_object_masks is not None:
            return self._all_object_masks[frame_id, cam_idx]
        else:
            return np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)

    def get_all_colors(self):
        """返回所有颜色图，shape: (num_frames, num_cams, H, W, 3)"""
        return self._all_colors

    def get_all_depths(self):
        """返回所有深度图，shape: (num_frames, num_cams, H, W)"""
        return self._all_depths

    def get_all_masks(self, object_idx=None, kernel_size=1):
        """
        返回所有mask，shape: (num_frames, num_cams, H, W)
        如果object_idx不为None，则返回指定object的mask（已二值化和腐蚀）
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
        返回所有object mask，shape: (num_frames, num_cams, H, W)
        如果没有object mask，则返回全0的mask
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
        valid_serials = []
        for serial in self._rs_serials:
            if (self._seg_folder / serial / "mask" / "mask_000000.png").exists():
                valid_serials.append(serial)
        return valid_serials

    def get_seg_color_index_map(self):
        color_index_map = {color.rgb: idx for idx, color in enumerate(HO_CAP_SEG_COLOR)}
        return color_index_map

    def get_object_seg_color(self, object_id):
        idx = self._object_ids.index(object_id) + 1
        color = HO_CAP_SEG_COLOR[idx].rgb
        return color

    def get_mano_seg_color(self, side):
        idx = 5 if side == "right" else 6
        color = HAND_COLORS[idx].rgb
        return color

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def rs_serials(self):
        return self._rs_serials

    @property
    def rs_width(self):
        return self._rs_width

    @property
    def rs_height(self):
        return self._rs_height

    @property
    def hl_serial(self):
        return self._hl_serial

    @property
    def hl_width(self):
        return self._hl_width

    @property
    def hl_height(self):
        return self._hl_height

    @property
    def rs_Ks(self):
        return self._rs_Ks

    @property
    def hl_K(self):
        return self._hl_K

    @property
    def extr2world(self):
        return self._extr2world

    @property
    def extr2world_inv(self):
        return self._extr2world_inv

    @property
    def mano_sides(self):
        return self._mano_sides

    @property
    def mano_beta(self):
        return self._mano_beta

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def object_textured_files(self):
        return self._object_textured_files

    @property
    def object_cleaned_files(self):
        return self._object_cleaned_files

    @property
    def x_threshold(self):
        return self._thresholds[0:2]
        # return tuple(self._thresholds["x"])

    @property
    def y_threshold(self):
        return self._thresholds[2:4]
        # return tuple(self._thresholds["y"])

    @property
    def z_threshold(self):
        return self._thresholds[4:6]
        # return tuple(self._thresholds["z"])
