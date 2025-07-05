import open3d as o3d

def view_ply(ply_path):
    # 读取 PLY 文件
    mesh = o3d.io.read_triangle_mesh(ply_path)

    # 检查是否包含法线和颜色，如果没有可以计算
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        print("[Warning] Mesh has no vertex colors.")

    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])

    # 可视化
    # o3d.visualization.draw_geometries([mesh],
    #                                   window_name="PLY Viewer",
    #                                   width=800,
    #                                   height=600,
    #                                   mesh_show_back_face=True)

# 示例调用
if __name__ == "__main__":
    # ply_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_195123/processed/fd_pose_solver/debug/wooden_spoon/scene_complete.ply"  # 修改为你的 ply 路径
    # 是一个中间凹陷的勺子
    # ply_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/debug_outputs/frame_0_filtered.ply"
    # ply_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/debug_outputs/frame_3_raw.ply"
    # ply_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250626_203130/processed/fd_pose_solver/debug/wooden_spoon/scene_complete.ply"
    # optimize pose
    # ply_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250701_012148/processed/object_pose_solver/dpts/dpts_000000.ply"
    ply_path = "/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250701_012148/processed/object_pose_solver/dpts/dpts_000000.ply"

    # tarun's debug
    # ply_path = "/home/wys/learning-compliant/crq_ws/tool_use_benchmark/FoundationPose/3/scene_raw.ply"
    view_ply(ply_path)
