import yaml
import numpy as np
from pathlib import Path

# 模拟相机序列号：你可以改成真实的相机序列号
CAMERA_SERIALS = ["00", "01", "02", "03",
                  "04", "05", "06", "07"]

def load_camera_params(file_path):
    """ Load camera parameters from a YAML file and convert to required format """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # data is a list
    camera_params = []
    for cam in data:
        transformation = np.array(cam['transformation']).astype(np.float32)
        color_intrinsic_matrix = np.array(cam['color_intrinsic_matrix']).astype(np.float32)
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
    extrinsics_file = calibration_folder / extrinsics_folder / "extrinsics.yaml"

    
    calibration_folder.mkdir(parents=True, exist_ok=True)
    intrinsics_folder.mkdir(parents=True, exist_ok=True)
    extrinsics_folder.mkdir(parents=True, exist_ok=True)

    # === Save extrinsics ===
    extrinsics_dict = {}
    for serial, T in zip(CAMERA_SERIALS, extrinsics):
        T_3x4 = T[:3, :4]  # 3x4 matrix
        extrinsics_dict[serial] = T_3x4.flatten().tolist()

    extrinsics_dict["tag_0"] = np.eye(4)[:3].flatten().tolist()
    extrinsics_dict["tag_1"] = np.eye(4)[:3].flatten().tolist()

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
    camera_params_file = "/home/wys/learning-compliant/crq_ws/reconstructed_cameras_scaled.yaml"

    # 输出路径
    calib_root = Path("/home/wys/learning-compliant/crq_ws/HO-Cap-Annotation/my_dataset")

    # 读取
    camera_params = load_camera_params(camera_params_file)
    extrinsics = [p["transformation"] for p in camera_params]
    intrinsics = [p["color_intrinsic_matrix"] for p in camera_params]

    # 保存
    save_calibration(extrinsics, intrinsics, calib_root)
