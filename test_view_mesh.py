import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh
import pyrender
import numpy as np
from PIL import Image

def render_mesh(mesh, save_path, image_size=(512, 512), angle_deg=45):
    # 创建一个场景
    scene = pyrender.Scene()

    # 将Trimesh转为Pyrender的Mesh
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh)

    # 添加摄像头
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = trimesh.transformations.rotation_matrix(
        np.radians(angle_deg), [0, 1, 0], point=[0, 0, 0])
    scene.add(camera, pose=cam_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)

    # 渲染
    r = pyrender.OffscreenRenderer(*image_size)
    color, _ = r.render(scene)
    r.delete()

    # 保存图像
    Image.fromarray(color).save(save_path)
    print(f"[INFO] Saved rendered image to: {save_path}")

# 替换成你的数据加载器和 object_idx
object_idx = 0
from hocap_annotation.loaders.hocap_loader import HOCapLoader
data_loader = HOCapLoader("/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/subject_5/20250624_021150")

# 加载 mesh
textured_mesh = trimesh.load(data_loader.object_textured_files[object_idx])
cleaned_mesh = trimesh.load(data_loader.object_cleaned_files[object_idx])

# 创建输出文件夹
output_dir = "render_output"
os.makedirs(output_dir, exist_ok=True)

# 渲染并保存
render_mesh(textured_mesh, os.path.join(output_dir, "textured_mesh.png"))
render_mesh(cleaned_mesh, os.path.join(output_dir, "cleaned_mesh.png"))
