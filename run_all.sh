#!/bin/bash

# # 运行第一个序列的处理
# python tools/04-1-1_fd_pose_solver_prep.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003042 --object_idx 1

# python tools/04-1-1_fd_pose_solver_prep.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003328 --object_idx 1

# python tools/04-1-1_fd_pose_solver_prep.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250702_003757 --object_idx 1

# # 运行第一个序列的合并
# python tools/04-2_fd_pose_merger.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003042

# # # 可视化第一个序列
# python debug/visualize_ob_in_world.py --data_path test_1/20250702_003042

# # # 运行第二个序列的合并
# python tools/04-2_fd_pose_merger.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003328

# # # 可视化第二个序列
# python debug/visualize_ob_in_world.py --data_path test_1/20250702_003328

# # 运行第三个序列的合并
# python tools/04-2_fd_pose_merger.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250702_003757

# # 可视化第三个序列
# python debug/visualize_ob_in_world.py --data_path test_2/20250702_003757 --tool_name wooden_spoon

# ./run_local.sh --sequence_name test_1/20250702_003042 --object_idx 1 --tool_name blue_scooper --output_idx 1 --uuid mask_depth >> log.txt 2>&1
# ./run_local.sh --sequence_name test_1/20250702_003328 --object_idx 1 --tool_name blue_scooper --output_idx 1 --uuid mask_depth >> log.txt 2>&1
# ./run_local.sh --sequence_name test_2/20250702_003757 --object_idx 1 --tool_name wooden_spoon --output_idx 2 >> log.txt 2>&1
# ./run_local.sh --sequence_name test_2/20250630_165212 --object_idx 1 --tool_name wooden_spoon --output_idx 2 >> log.txt 2>&1
# ./run_detection.sh --sequence_name test_3/20250702_234933 --object_idx 1 --tool_name squeegee --output_idx 2 >> log.txt 2>&1

./run_local.sh --sequence_name squeegee_1/20250704_151206 --tool_name squeegee --object_idx 1 --output_idx 1 --uuid mask_depth >> log.txt 2>&1

./run_local.sh --sequence_name blue_scooper_1/20250704_172530 --tool_name blue_scooper --object_idx 1  --uuid mask_depth_and_object >> log.txt 2>&1

./run_local.sh --sequence_name wooden_spoon_1/20250704_202115 --tool_name wooden_spoon --object_idx 1  --uuid re_test >> log.txt 2>&1

./run_local.sh --sequence_name blue_scooper_1/20250704_202733 --tool_name blue_scooper --object_idx 1  --uuid mask_depth_and_object >> log.txt 2>&1

./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid mask_depth >> log.txt 2>&1

# 运行目标姿态求解
# python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003042

# python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003328

# python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250702_003757

