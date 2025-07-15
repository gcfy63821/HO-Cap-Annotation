import numpy as np
import trimesh
import pyrender
from pyrender.constants import RenderFlags
from ..utils import NUM_MANO_VERTS, get_logger
import cv2
cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class OffscreenRenderer:
    def __init__(self, znear=0.01, zfar=10.0) -> None:
        self._logger = get_logger(__class__.__name__, "DEBUG")
        self._znear = znear
        self._zfar = zfar
        self._cam_nodes = {}
        self._pyr_meshes = {}
        self._seg_colors = {}

    def __del__(self):
        self.clear_cameras()
        self.clear_meshes()

    def add_camera(self, cam_K, name):
        """Add a camera to the scene with its intrinsic matrix `cam_K`."""
        if name in self._cam_nodes:
            self._logger.warning(f"Camera {name} already exists. Overwriting...")
        self._cam_nodes[name] = pyrender.Node(
            name=name,
            camera=pyrender.IntrinsicsCamera(
                fx=cam_K[0, 0],
                fy=cam_K[1, 1],
                cx=cam_K[0, 2],
                cy=cam_K[1, 2],
                znear=self._znear,
                zfar=self._zfar,
            ),
            light=pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0),
        )

    def add_mesh(self, mesh, name, seg_color=(0, 0, 0)):
        """Add a mesh to the scene."""
        if isinstance(mesh, trimesh.Trimesh):
            pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
        elif isinstance(mesh, str):
            pyr_mesh = pyrender.Mesh.from_trimesh(trimesh.load(mesh))
        else:
            raise ValueError("Invalid mesh type! Must be trimesh or filepath.")
        self._pyr_meshes[name] = pyr_mesh
        self._seg_colors[name] = seg_color

    def remove_camera(self, name):
        """Remove a camera from the scene."""
        if name in self._cam_nodes:
            del self._cam_nodes[name]
        else:
            self._logger.warning(f"Camera {name} not found!!!")

    def remove_mesh(self, name):
        """Remove a mesh from the scene."""
        if name in self._pyr_meshes:
            del self._seg_colors[name]
            del self._pyr_meshes[name]
        else:
            self._logger.warning(f"Mesh {name} not found!!!")

    def clear_cameras(self):
        """Remove all cameras from the scene."""
        self._cam_nodes.clear()

    def clear_meshes(self):
        """Remove all meshes from the scene."""
        self._seg_colors.clear()
        self._pyr_meshes.clear()

    def _is_valid_pose(self, pose):
        """Check if the pose is a valid 4x4 transformation matrix."""
        if pose is None or not isinstance(pose, np.ndarray):
            return False
        return pose.shape == (4, 4) and not np.all(pose == -1)

    def _convert_to_glcam_poses(self, cam_poses):
        """Convert CV camera poses to GL camera space."""
        if isinstance(cam_poses, list):
            return [pose @ cvcam_in_glcam for pose in cam_poses]
        return cam_poses @ cvcam_in_glcam

    def _add_nodes_to_scene(
        self, scene, parent_node, node_dict, node_names, node_poses, seg_colors
    ):
        """Add nodes to the scene with poses and segmentation colors."""
        seg_node_map = {}
        if isinstance(node_names, list):
            if len(node_names) != len(node_poses):
                print("DEBUG: node names and poses:",node_names, node_poses)
                print("DEBUG length:", len(node_names), len(node_poses))
                raise ValueError("Mismatch between node_names and node_poses length")
            for name, pose in zip(node_names, node_poses):
                if name in node_dict and self._is_valid_pose(pose):
                    node = node_dict[name]
                    scene.add_node(node, parent_node=parent_node)
                    scene.set_pose(node, pose)
                    if name in seg_colors:
                        seg_node_map[node] = seg_colors[name]
        elif node_names in node_dict and self._is_valid_pose(node_poses):
            node = node_dict[node_names]
            scene.add_node(node, parent_node=parent_node)
            scene.set_pose(node, poses)
            if node_names in seg_colors:
                seg_node_map[node] = seg_colors[node_names]
        return seg_node_map

    def _add_mano_meshes_to_scene(
        self, scene, parent_node, mano_verts, mano_faces, mano_colors, wireframe=False
    ):
        """Add MANO mesh nodes to the scene."""
        seg_node_map = {}
        mano_meshes = trimesh.Trimesh(
            vertices=mano_verts, faces=mano_faces, process=False
        ).split(only_watertight=False)
        if len(mano_meshes) != len(mano_colors):
            raise ValueError("Mismatch between mano_meshes and mano_colors length!")
        for idx, (mano_mesh, mano_color) in enumerate(zip(mano_meshes, mano_colors)):
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=1.0,
                alphaMode="OPAQUE",
                doubleSided=True,
                baseColorFactor=mano_color,
            )
            mano_node = pyrender.Node(
                name=f"mano_{idx}",
                mesh=pyrender.Mesh.from_trimesh(mano_mesh, material=material),
            )
            scene.add_node(mano_node, parent_node=parent_node)
            seg_node_map[mano_node] = np.array(mano_color) * 255
            if wireframe:
                mano_mesh_wf = pyrender.Mesh.from_trimesh(mano_mesh, wireframe=True)
                mano_mesh_wf.primitives[0].material.baseColorFactor = [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                scene.add_node(
                    pyrender.Node(name=f"mano_{idx}_wf", mesh=mano_mesh_wf),
                    parent_node=parent_node,
                )
        return seg_node_map

    def _create_scene(
        self,
        cam_names=None,
        cam_poses=None,
        mesh_names=None,
        mesh_poses=None,
        mano_vertices=None,
        mano_faces=None,
        mano_colors=None,
        bg_color=[0, 0, 0, 0],
        ambient_light=[1.0, 1.0, 1.0],
        seg_obj=False,
        seg_mano=False,
    ):
        """Create a scene with the provided camera and mesh information."""
        seg_node_map = {}
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
        world_node = scene.add_node(pyrender.Node(name="world"))
        # Add camera nodes to the scene
        if cam_names is not None and cam_poses is not None:
            glcam_poses = self._convert_to_glcam_poses(cam_poses)
            self._add_nodes_to_scene(
                scene, world_node, self._cam_nodes, cam_names, glcam_poses, {}
            )
        # Add mesh nodes to the scene
        if mesh_names is not None and mesh_poses is not None:
            mesh_nodes = {
                name: pyrender.Node(name=name, mesh=self._pyr_meshes[name])
                for name in mesh_names
            }
            seg_node_map_o = self._add_nodes_to_scene(
                scene,
                world_node,
                mesh_nodes,
                mesh_names,
                mesh_poses,
                self._seg_colors,
            )
            if not seg_obj:
                for node in seg_node_map_o:
                    seg_node_map_o[node] = (0, 0, 0)
            seg_node_map.update(seg_node_map_o)
        # Add MANO hand nodes to the scene
        if mano_vertices is not None and mano_faces is not None:
            seg_node_map_m = self._add_mano_meshes_to_scene(
                scene, world_node, mano_vertices, mano_faces, mano_colors
            )
            if not seg_mano:
                for node in seg_node_map_m:
                    seg_node_map_m[node] = (0, 0, 0)
            seg_node_map.update(seg_node_map_m)
        return scene, seg_node_map

    def _render_scene(
        self,
        width,
        height,
        cam_names,
        cam_poses,
        mesh_names,
        mesh_poses,
        mano_vertices,
        mano_faces,
        mano_colors,
        bg_color,
        ambient_light=[1.0, 1.0, 1.0],
        render_flags=0,
        seg_obj=False,
        seg_mano=False,
    ):
        """Render the scene with the provided camera and mesh information."""

        r = pyrender.OffscreenRenderer(width, height)
        try:
            scene, seg_node_map = self._create_scene(
                cam_names=cam_names,
                cam_poses=cam_poses,
                mesh_names=mesh_names,
                mesh_poses=mesh_poses,
                mano_vertices=mano_vertices,
                mano_faces=mano_faces,
                mano_colors=mano_colors,
                bg_color=bg_color,
                ambient_light=ambient_light,
                seg_obj=seg_obj,
                seg_mano=seg_mano,
            )
            # print(f"[DEBUG] Scene has {len(scene.mesh_nodes)} mesh nodes, {len(scene.camera_nodes)} camera nodes")
            # if isinstance(cam_names, list):
            #     for cam_name in cam_names:
            #         assert cam_name in self._cam_nodes, f"Missing camera node for: {cam_name}"
            #         print(f"[DEBUG] Rendering from cam: {cam_name}")
            # 添加可见光源
            # light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            # scene.add(light, pose=np.eye(4))  # 光源从默认位置照射

            # 打印相机与 mesh 位姿检查朝向
            # for name, pose in zip(cam_names, cam_poses):
            #     print(f"[DEBUG] cam {name} pos: {pose[:3, 3]}")
            # for i, pose in enumerate(mesh_poses):
            #     print(f"[DEBUG] mesh {i} pos: {pose[:3, 3]}")

            # # 保存可视化
            # pyrender.Viewer(scene, use_raymond_lighting=True)  # 若你有图形界面支持


            colors, depths = [], []
            if isinstance(cam_names, list):
                for cam_name in cam_names:
                    scene.main_camera_node = self._cam_nodes[cam_name]
                    color, depth = r.render(scene, render_flags, seg_node_map)
                    colors.append(color)
                    depths.append(depth)
            else:
                scene.main_camera_node = self._cam_nodes[cam_names]
                colors, depths = r.render(scene, render_flags, seg_node_map)
            # print(f"[DEBUG] Rendered image stats - mean: {color.mean()}, min: {color.min()}, max: {color.max()}")
            # self._logger.debug(f"Rendered image stats - mean: {color.mean()}, min: {color.min()}, max: {color.max()}")
            cv2.imwrite(f"debug_render/cam_{cam_name}.png", color[..., ::-1])  # BGR save

        finally:
            r.delete()

        return colors, depths

    def get_render_colors(
        self,
        width,
        height,
        cam_names,
        cam_poses,
        mesh_names=None,
        mesh_poses=None,
        mano_vertices=None,
        mano_faces=None,
        mano_colors=None,
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.3, 0.3, 0.3],
    ):
        """
        Render colors from the scene based on the provided camera and mesh information.

        Returns:
        list of arrays: Rendered colors for each camera view.
        """

        colors, _ = self._render_scene(
            width,
            height,
            cam_names,
            cam_poses,
            mesh_names,
            mesh_poses,
            mano_vertices,
            mano_faces,
            mano_colors,
            bg_color,
        )
        # print(f"#rendered images: {len(colors)}")
        return colors

    def get_render_depths(
        self,
        width,
        height,
        cam_names,
        cam_poses,
        mesh_names=None,
        mesh_poses=None,
        mano_vertices=None,
        mano_faces=None,
        mano_colors=None,
        bg_color=[0, 0, 0, 0],
    ):
        """
        Render depths from the scene based on the provided camera and mesh information.

        Returns:
        list of arrays: Rendered depths for each camera view.
        """
        _, depths = self._render_scene(
            width,
            height,
            cam_names,
            cam_poses,
            mesh_names,
            mesh_poses,
            mano_vertices,
            mano_faces,
            mano_colors,
            bg_color,
        )
        return depths

    def get_render_segs(
        self,
        width,
        height,
        cam_names,
        cam_poses,
        mesh_names=None,
        mesh_poses=None,
        mano_vertices=None,
        mano_faces=None,
        mano_colors=None,
        bg_color=[0, 0, 0, 0],
        seg_obj=True,
        seg_mano=True,
    ):
        """
        Render segmentations from the scene based on the provided camera and mesh information.

        Returns:
        list of arrays: Rendered segmentations for each camera view.
        """
        segs, _ = self._render_scene(
            width=width,
            height=height,
            cam_names=cam_names,
            cam_poses=cam_poses,
            mesh_names=mesh_names,
            mesh_poses=mesh_poses,
            mano_vertices=mano_vertices,
            mano_faces=mano_faces,
            mano_colors=mano_colors,
            bg_color=bg_color,
            render_flags=RenderFlags.SEG,
            seg_obj=seg_obj,
            seg_mano=seg_mano,
        )
        return segs
