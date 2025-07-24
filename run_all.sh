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
# bad spoon
# ./run_local.sh --sequence_name test_2/20250702_003757 --object_idx 1 --tool_name wooden_spoon --uuid mask_depth >> log.txt 2>&1


# ./run_local.sh --sequence_name test_2/20250630_165212 --object_idx 1 --tool_name wooden_spoon  --uuid 0711 >> log.txt 2>&1
# # # ./run_detection.sh --sequence_name test_3/20250702_234933 --object_idx 1 --tool_name squeegee --output_idx 2 >> log.txt 2>&1

# ./run_local.sh --sequence_name squeegee_1/20250704_151206 --tool_name squeegee --object_idx 1  --uuid 0711 >> log.txt 2>&1

# ./run_local.sh --sequence_name blue_scooper_1/20250704_172530 --tool_name blue_scooper --object_idx 1  --uuid 0711 >> log.txt 2>&1

# # ./run_local.sh --sequence_name wooden_spoon_1/20250704_202115 --tool_name wooden_spoon --object_idx 1  --uuid re_test >> log.txt 2>&1

# ./run_local.sh --sequence_name blue_scooper_1/20250704_202733 --tool_name blue_scooper --object_idx 1  --uuid 0711  >> log.txt 2>&1

# ./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid 0711  >> log.txt 2>&1

# ./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name purple_plate --object_idx 2  --uuid 0711 --optimize 1 >> log.txt 2>&1
# # new extrinsics
# ./run_local.sh --sequence_name wooden_spoon_1/20250705_210948 --tool_name wooden_spoon --object_idx 1  --uuid updated_ext >> log.txt 2>&1

# ./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name purple_plate --object_idx 2  --uuid 0712_plate_20iter --track_refine_iter 20 >> log.txt 2>&1

# ./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name purple_plate --object_idx 2  --uuid 0712_plate_40iter --track_refine_iter 40 >> log.txt 2>&1

# ./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid 0712_pestle_40iter --track_refine_iter 40 >> log.txt 2>&1

# ./run_local.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid 0712_pestle_20iter --track_refine_iter 20 >> log.txt 2>&1

# ./run_whole_process.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid 0714_whole_process --hand 1>> log.txt 2>&1

# ./run_optimize.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid 0714_whole_process --hand 1 --optimize 1>> log.txt 2>&1

# ./run_whole_process.sh --sequence_name blue_scooper_1/20250704_202733 --tool_name blue_scooper  --uuid 0716 --hand 1 --optimize 1  >> log.txt 2>&1
# ./run_whole_process.sh --sequence_name blue_scooper_1/20250704_172530 --tool_name blue_scooper  --uuid 0716 --hand 1 --optimize 1  >> log.txt 2>&1
# ./run_whole_process.sh --sequence_name pestle_1/20250703_132710 --tool_name pestle --object_idx 1  --uuid 0716 --hand 1 --optimize 1>> log.txt 2>&1
# ./run_whole_process.sh --sequence_name wooden_spoon_1/20250705_210948 --tool_name wooden_spoon --object_idx 1  --uuid 0716_newext  --optimize 1 
# ./run_whole_process.sh --sequence_name wooden_spoon_1/20250704_202115 --tool_name wooden_spoon --object_idx 1  --uuid 0719_test  --optimize 1 --hand 1

# ./run_hand.sh --sequence_name squeegee_1/20250704_151206 --tool_name squeegee --object_idx 1  --uuid 0716_new_hand  --optimize 1 


# ./run_hand.sh --sequence_name videos_0713/20250721_004007 --tool_name green_straw --object_idx 1  --uuid 0721_hand_nocrop  --optimize 1 --crop_view 0 >> log.txt 2>&1
# ./run_hand.sh --sequence_name videos_0713/20250721_004007 --tool_name green_straw --object_idx 1  --uuid 0721_hand_crop  --optimize 1 --crop_view 1 >> log.txt 2>&1

# ./run_separate.sh --sequence_name videos_0713/20250721_193540 --tool_name green_straw --object_idx 1  --uuid 0721_separate  --optimize 1 --crop_view 0 >> log.txt 2>&1
# ./run_separate.sh --sequence_name videos_0713/20250722_020449 --tool_name green_straw --object_idx 1  --uuid 0722_separate_crop_choose  --crop_view 1 >> log.txt 2>&1
# ./run_separate.sh --sequence_name videos_0713/20250722_020449 --tool_name green_straw --object_idx 1  --uuid 0722_separate_choose_view  --crop_view 0 >> log.txt 2>&1

# ./run_separate.sh --sequence_name videos_0713/coffee_1_1 --tool_name green_straw --object_idx 1  --uuid 0722_separate_choose_view  --crop_view 0 >> log.txt 2>&1
# ./run_hand.sh --sequence_name videos_0713/coffee_1_1 --tool_name green_straw --object_idx 1  --uuid 0722_separate_choose_view  --crop_view 0 >> log.txt 2>&1

# ./run_separate.sh --sequence_name videos_0713/coffee_1_1 --tool_name green_straw --object_idx 1  --uuid 0723new_rot_1_trans_0.03 --rot_thresh 1 --trans_thresh 0.03 --crop_view 0 --track_refine_iter 20 >> log.txt 2>&1
# ./run_separate.sh --sequence_name videos_0713/coffee_1_1 --tool_name green_straw --object_idx 1  --uuid 0723new_rot_0.5_trans_0.02 --rot_thresh 0.5 --trans_thresh 0.02 --crop_view 0 --track_refine_iter 20 >> log.txt 2>&1
# ./run_separate.sh --sequence_name videos_0713/coffee_1_1 --tool_name green_straw --object_idx 1  --uuid 0723new_rot_0.25_trans_0.02 --rot_thresh 0.25 --trans_thresh 0.02 --crop_view 0 --track_refine_iter 20 >> log.txt 2>&1
./run_separate.sh --sequence_name videos_0713/coffee_1_1 --tool_name green_straw --object_idx 1  --uuid test0724_rot_0.25_trans_0.03 --rot_thresh 0.25 --trans_thresh 0.03 --crop_view 0 --track_refine_iter 10 >> log.txt 2>&1
./run_separate.sh --sequence_name videos_0713/peeler_0_1 --tool_name peeler --object_idx 1  --uuid test0724_rot_0.25_trans_0.03 --rot_thresh 0.25 --trans_thresh 0.03 --crop_view 0 --track_refine_iter 10 >> log.txt 2>&1
./run_separate.sh --sequence_name videos_0713/wooden_brush_1_1 --tool_name wooden_brush --object_idx 1  --uuid test0724_rot_0.25_trans_0.03 --rot_thresh 0.25 --trans_thresh 0.03 --crop_view 0 --track_refine_iter 10 >> log.txt 2>&1
./run_separate.sh --sequence_name videos_0713/wooden_spoon_0_1 --tool_name wooden_spoon --object_idx 1  --uuid test0724_rot_0.25_trans_0.03 --rot_thresh 0.25 --trans_thresh 0.03 --crop_view 0 --track_refine_iter 10 >> log.txt 2>&1


# 运行目标姿态求解
# python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003042

# python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_1/20250702_003328

# python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/test_2/20250702_003757

