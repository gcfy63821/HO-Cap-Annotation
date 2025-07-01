import trimesh

def simplify_and_scale_obj(
    input_path,
    output_path,
    scale_factor=0.001,
    target_fraction=0.25
):
    # 加载模型
    mesh = trimesh.load(input_path, process=True)

    print(f"[INFO] 原始顶点数: {len(mesh.vertices)}")
    print(f"[INFO] 原始面数: {len(mesh.faces)}")

    # 缩放单位（从 mm 转为 m）
    mesh.apply_scale(scale_factor)

    # 网格简化（面数量减少）
    target_faces = int(len(mesh.faces) * target_fraction)
    mesh_simplified = mesh.simplify_quadratic_decimation(target_faces)

    print(f"[INFO] 简化后顶点数: {len(mesh_simplified.vertices)}")
    print(f"[INFO] 简化后面数: {len(mesh_simplified.faces)}")

    # 保存为新的 obj 文件
    mesh_simplified.export(output_path)
    print(f"[INFO] 已保存简化 + 缩放模型至：{output_path}")

if __name__ == "__main__":
    # 替换为你的路径
    input_obj_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/models/wooden_spoon/cleaned_mesh_10000.obj"
    output_obj_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/models/wooden_spoon/cleaned_mesh_10000.obj"
    
    simplify_and_scale_obj(
        input_path=input_obj_path,
        output_path=output_obj_path,
        scale_factor=0.001,       # mm -> m
        target_fraction=0.25      # 仅保留 25% 面
    )
