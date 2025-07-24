import trimesh

def inspect_obj_model(obj_path):
    # 加载模型
    mesh = trimesh.load(obj_path, process=True)

    # 输出关键信息
    print("=== OBJ 模型信息 ===")
    print(f"文件: {obj_path}")
    print(f"顶点数: {len(mesh.vertices)}")
    print(f"面数: {len(mesh.faces)}")
    print(f"单位体积: {mesh.volume:.6f}")
    print(f"表面积: {mesh.area:.6f}")
    print(f"轴对齐边界盒 AABB:\n{mesh.bounds}")
    print(f"有向边界盒 OBB（center, extents, transform）:")
    obb = mesh.bounding_box_oriented
    print(f"  中心: {obb.centroid}")
    print(f"  尺寸 (XYZ extents): {obb.primitive.extents}")
    print(f"  变换矩阵:\n{obb.primitive.transform}")
    print(f"重心 (center of mass): {mesh.center_mass}")
    print(f"质心 (centroid): {mesh.centroid}")

    # 如果需要，也可以显示模型
    # mesh.show()

if __name__ == "__main__":
    # 替换为你的 .obj 文件路径
    # obj_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/datasets/models/G01_1/cleaned_mesh_10000.obj"
    obj_path = "/home/wys/learning-compliant/crq_ws/data/mesh_files/wooden_spoon.obj"
    inspect_obj_model(obj_path)
