import trimesh
import numpy as np
import copy
import cv2
from PIL import Image
import os
from pathlib import Path
from trimesh.visual.texture import SimpleMaterial, TextureVisuals
from trimesh.exchange.obj import export_obj
import shutil
import argparse

def save_textured_mesh_as_obj(obj_mesh, texture_path, output_prefix):
    """
    obj_mesh: trimesh.Trimesh对象
    texture_path: 纹理图片路径
    output_prefix: 输出文件前缀（不带扩展名）
    """
    output_prefix = Path(output_prefix)
    if not hasattr(obj_mesh.visual, 'uv') or obj_mesh.visual.uv is None:
        raise ValueError("Mesh does not contain UV coordinates. Cannot apply texture.")

    image = Image.open(texture_path)
    material_name = "material_0"
    mtl_filename = output_prefix.with_suffix(".mtl").name
    texture_filename = Path(texture_path).name
    material = SimpleMaterial(image=texture_filename, name=material_name)
    obj_mesh.visual = TextureVisuals(uv=obj_mesh.visual.uv, image=image, material=material)
    obj_str = export_obj(obj_mesh, include_texture=True)
    obj_file = output_prefix.with_suffix(".obj")

    # with open(obj_file, "w") as f:
    #     f.write(f"mtllib {mtl_filename}\n")
    #     f.write(obj_str)
    mtl_file = output_prefix.with_suffix(".mtl")
    with open(mtl_file, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Ka 1.000 1.000 1.000\n")
        f.write(f"Kd 1.000 1.000 1.000\n")
        f.write(f"Ks 0.000 0.000 0.000\n")
        f.write(f"d 1.0\n")
        f.write(f"illum 2\n")
        f.write(f"map_Kd {texture_filename}\n")
    shutil.copy(texture_path, obj_file.parent / texture_filename)
    print(f"[✓] Exported textured mesh to {obj_file} with .mtl and texture.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool_name", type=str, required=True, help="工具名称")
    # parser.add_argument("--mesh_dir", type=str, default="/home/wys/learning-compliant/crq_ws/data/mesh/decim_mesh_files", help="mesh目录")

    parser.add_argument("--mesh_dir", type=str, default="/home/wys/learning-compliant/crq_ws/data/mesh_files", help="mesh目录")
    parser.add_argument("--texture_dir", type=str, default="/home/wys/learning-compliant/crq_ws/data/mesh/textures/textures", help="纹理图片目录")
    parser.add_argument("--output_dir", type=str, default="/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/models", help="输出目录")
    args = parser.parse_args()

    tool_name = args.tool_name
    mesh_path = f"{args.mesh_dir}/{tool_name}.obj"
    texture_img_path = f"{args.texture_dir}/{tool_name}.jpg"
    output_path = f"{args.output_dir}/{tool_name}/cleaned_mesh_10000.obj"
    textured_path = f"{args.output_dir}/{tool_name}/textured_mesh.obj"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh = trimesh.load(mesh_path, process=True)
    texture = cv2.imread(texture_img_path)
    im = Image.open(texture_img_path)
    uv = mesh.visual.uv
    material = trimesh.visual.texture.SimpleMaterial(image=im)
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
    mesh.visual = color_visuals

    # 缩放
    mesh.vertices *= 0.001

    # 应用oriented bounds变换
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    mesh.apply_transform(to_origin)

    # 保存mesh (确保文件格式支持纹理)
    mesh.export(output_path, file_type='obj')
    mesh.export(textured_path, file_type='obj')

    print(f"Mesh loaded from {mesh_path} and saved to {output_path}")

    # 重新加载保存后的mesh
    new_mesh = trimesh.load(output_path, process=True, file_type='obj')
    print(len(new_mesh.vertices))

    # if len(new_mesh.vertices) > 200000: # fix
    #     new_mesh = new_mesh.simplify_quadric_decimation(0.8)
    

    # 检查UV坐标
    if not hasattr(new_mesh.visual, 'uv') or new_mesh.visual.uv is None:
        raise ValueError("Mesh does not contain UV coordinates. Cannot apply texture.")

    # 检查纹理图片是否存在，存在则导出带贴图的mesh
    if os.path.exists(texture_img_path):
        save_textured_mesh_as_obj(mesh, texture_img_path, textured_path[:-4])  # 去掉.obj后缀
    else:
        print(f"[WARN] Texture image not found: {texture_img_path}, skip textured mesh export.")


