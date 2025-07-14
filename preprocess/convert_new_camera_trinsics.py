# yaml file format:
'''
- camera_id: 0
  color_intrinsic_matrix:
  - - 606.386474609375
    - 0.0
    - 322.0896911621094
  - - 0.0
    - 606.250732421875
    - 244.8662567138672
  - - 0.0
    - 0.0
    - 1.0
  depth_intrinsic_matrix:
  - - 388.8372802734375
    - 0.0
    - 328.7183837890625
  - - 0.0
    - 388.8372802734375
    - 237.40037536621094
  - - 0.0
    - 0.0
    - 1.0
  serial_number: '244222072252'
  transformation:
  - - -0.6214960813522339
    - 0.42471954226493847
    - -0.6582976579666138
    - 0.5046114321656868
  - - 0.7554557919502258
    - 0.10241666436195371
    - -0.6471455693244934
    - 0.5569536332979523
  - - -0.20743465423583993
    - -0.8995132446289062
    - -0.38450843095779424
    - 0.36043259618305384
  - - 0.0
    - 0.0
    - 0.0
    - 1.0
- camera_id: 1
  color_intrinsic_matrix:
  - - 608.4426879882812
  ...
'''

import yaml
import numpy as np
from pathlib import Path


CAMERA_SERIALS = ["00", "01", "02", "03",
                  "04", "05", "06", "07"]

def load_camera_params(file_path):
    """ Load camera parameters from a YAML file and convert to required format """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # data is a list
    camera_params = []
    for idx, cam in enumerate(data):
        transformation = np.array(cam['transformation']).astype(np.float32)
        color_intrinsic_matrix = np.array(cam['color_intrinsic_matrix']).astype(np.float32)
        # 保证顺序与00-07一致
        camera_params.append({
            'transformation': transformation,
            'color_intrinsic_matrix': color_intrinsic_matrix
        })
    return camera_params

def save_calibration(extrinsics, intrinsics, calib_root):
    """ Save intrinsics per camera and all extrinsics in a single YAML file """

    calibration_folder = calib_root / "calibration"
    intrinsics_folder = calibration_folder / "intrinsics"
    extrinsics_folder = calibration_folder / "extrinsics"
    extrinsics_file = extrinsics_folder / "extrinsics_updated_0629.yaml"

    calibration_folder.mkdir(parents=True, exist_ok=True)
    intrinsics_folder.mkdir(parents=True, exist_ok=True)
    extrinsics_folder.mkdir(parents=True, exist_ok=True)

    # === Save extrinsics ===
    extrinsics_dict = {}
    T_base_world = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
            
    for serial, T in zip(CAMERA_SERIALS, extrinsics):
        T_base = T_base_world @ T  # World to camera transformation
        T_3x4 = T_base[:3, :4]  # Extract the 3x4 part
        extrinsics_dict[serial] = T_3x4.flatten().tolist()

    # extrinsics_dict["tag_0"] = np.eye(4)[:3].flatten().tolist()
    # extrinsics_dict["tag_1"] = np.eye(4)[:3].flatten().tolist()

    with open(extrinsics_file, 'w') as f:
        yaml.dump({"extrinsics": extrinsics_dict, "rs_master": CAMERA_SERIALS[4]}, f)

    # === Save intrinsics per camera ===
    for serial, K in zip(CAMERA_SERIALS, intrinsics):
        intrinsics_data = {
            "serial": serial,
            "color": {
                "width": 640,
                "height": 480,
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "ppx": float(K[0, 2]),
                "ppy": float(K[1, 2]),
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]  # placeholder
            },
            "depth": {
                "width": 640,
                "height": 480,
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "ppx": float(K[0, 2]),
                "ppy": float(K[1, 2]),
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]  # placeholder
            },
            "depth2color": [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        }

        with open(intrinsics_folder / f"{serial}.yaml", 'w') as f:
            yaml.dump(intrinsics_data, f)

    print(f"Calibration saved to: {calibration_folder}")

# 示例调用
if __name__ == "__main__":
    # 输入相机参数的路径
    camera_params_file = "/home/wys/learning-compliant/crq_ws/updated_extrinsics_0629.yaml"

    # 输出路径
    calib_root = Path("/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset")

    # 读取
    camera_params = load_camera_params(camera_params_file)
    extrinsics = [p["transformation"] for p in camera_params]
    intrinsics = [p["color_intrinsic_matrix"] for p in camera_params]

    # 保存
    save_calibration(extrinsics, intrinsics, calib_root)
    save_calibration(extrinsics, intrinsics, calib_root)
