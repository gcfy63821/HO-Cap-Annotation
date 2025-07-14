#!/bin/bash

python tools/05_mano_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/squeegee_1/20250704_151206 --debug

python tools/06_object_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/squeegee_1/20250704_151206 --debug

python tools/07_joint_pose_solver.py --sequence_folder /home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset/squeegee_1/20250704_151206 --debug