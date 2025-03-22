from .common_imports import *


def make_clean_folder(folder_path: Union[str, Path]) -> None:
    """Delete the folder if it exists and create a new one."""
    if Path(folder_path).is_dir():
        shutil.rmtree(str(folder_path))
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create folder '{folder_path}': {e}")


def copy_file(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Copy a file from the source path to the destination path."""
    if not Path(src_path).is_file():
        raise FileNotFoundError(f"Source file does not exist: '{src_path}'.")
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(str(src_path), str(dst_path))
    except OSError as e:
        raise OSError(f"Failed to copy file from '{src_path}' to '{dst_path}': {e}")


def copy_folder(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Copy a folder from the source path to the destination path."""
    if not Path(src_path).is_dir():
        raise FileNotFoundError(f"Source folder does not exist: '{src_path}'.")
    if Path(dst_path).is_dir():
        shutil.rmtree(str(dst_path))
    try:
        # Copy the source folder to the destination
        shutil.copytree(str(src_path), str(dst_path))
    except OSError as e:
        raise OSError(f"Failed to copy folder from '{src_path}' to '{dst_path}': {e}")


def delete_file(file_path: Union[str, Path]) -> None:
    """Delete a file if it exists."""
    _file_path = Path(file_path)

    # Check if the path is a file
    if _file_path.is_file():
        try:
            _file_path.unlink()
        except OSError as e:
            raise OSError(f"Failed to delete file '{_file_path}': {e}")


def delete_folder(folder_path: Union[str, Path]) -> None:
    """Delete a folder if it exists."""
    if Path(folder_path).is_dir():
        shutil.rmtree(str(folder_path))


def move_file(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Move a file from the source path to the destination path."""
    if not Path(src_path).is_file():
        raise FileNotFoundError(f"Source file does not exist: '{src_path}'.")
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_path), str(dst_path))


def move_folder(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Move a folder from the source path to the destination path."""
    if not Path(src_path).is_dir():
        raise FileNotFoundError(f"Source folder does not exist: '{src_path}'.")
    if Path(dst_path).is_dir():
        shutil.rmtree(str(dst_path))
    try:
        shutil.move(str(src_path), str(dst_path))
    except OSError as e:
        raise OSError(f"Failed to move folder from '{src_path}' to '{dst_path}': {e}")


def read_data_from_json(file_path: Union[str, Path]) -> Any:
    """Read data from a JSON file and return it."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(str(file_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON from {file_path}: {e}")


def write_data_to_json(file_path: Union[str, Path], data: Union[list, Dict]) -> None:
    """Write data to a JSON file."""
    try:
        with open(str(file_path), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
    except IOError as e:
        raise IOError(f"Failed to write JSON data to {file_path}: {e}")


def read_data_from_pickle(file_path: Union[str, Path]) -> Any:
    """Read data from a pickle file and return it."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(str(file_path), "rb") as f:
            return pkl.load(f)
    except pkl.UnpicklingError as e:
        raise ValueError(f"Error reading pickle file from {file_path}: {e}")


def write_data_to_pickle(file_path: Union[str, Path], data: Any) -> None:
    """Write data to a pickle file."""
    try:
        with open(str(file_path), "wb") as f:
            pkl.dump(data, f)
    except IOError as e:
        raise IOError(f"Failed to write pickle data to {file_path}: {e}")


def read_data_from_yaml(file_path: Union[str, Path]) -> Any:
    """Read data from a YAML file and return it."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(str(file_path), "r", encoding="utf-8") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading YAML file from {file_path}: {e}")


def write_data_to_yaml(file_path: Union[str, Path], data: Any) -> None:
    """Write data to a YAML file."""
    try:
        with open(str(file_path), "w", encoding="utf-8") as f:
            yaml.dump(data, f)
    except IOError as e:
        raise IOError(f"Failed to write YAML data to {file_path}: {e}")


def read_pose_from_txt(file_path: Union[str, Path]) -> np.ndarray:
    """Read a pose matrix from a text file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Pose file '{file_path}' does not exist.")
    try:
        pose = np.loadtxt(str(file_path), dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to load pose from '{file_path}': {e}")
    return pose


def write_pose_to_txt(
    pose_path: Union[str, Path], pose: np.ndarray, header: str = "", fmt: str = "%.8f"
) -> None:
    """Write a pose matrix to a text file."""
    try:
        np.savetxt(str(pose_path), pose, fmt=fmt, header=header)
    except Exception as e:
        raise ValueError(f"Failed to write pose to '{pose_path}': {e}")


def read_rgb_image(file_path: Union[str, Path]) -> np.ndarray:
    """Read an RGB image from the specified file path."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to load image from '{file_path}'.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_rgb_image(file_path: Union[str, Path], image: np.ndarray) -> None:
    """Write an RGB image to the specified file path."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")
    success = cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError(f"Failed to write RGB image to '{file_path}'.")


def read_depth_image(file_path: Union[str, Path], scale: float = 1.0) -> np.ndarray:
    """Read a depth image from the specified file path."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Depth image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"Failed to load depth image from '{file_path}'.")
    image = image.astype(np.float32) / scale
    return image


def write_depth_image(file_path: Union[str, Path], image: np.ndarray) -> None:
    """Write a depth image to the specified file path."""
    if image.dtype not in [np.uint16, np.uint8]:
        raise ValueError("Depth image must be of type uint16 or uint8.")
    success = cv2.imwrite(str(file_path), image)
    if not success:
        raise ValueError(f"Failed to write depth image to '{file_path}'.")


def read_mask_image(file_path: Union[str, Path]) -> np.ndarray:
    """Read a mask image from the specified file path."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Mask image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load mask image from '{file_path}'.")
    return image


def write_mask_image(file_path: Union[str, Path], image: np.ndarray) -> None:
    """Write a mask image to the specified file path."""
    success = cv2.imwrite(str(file_path), image)
    if not success:
        raise ValueError(f"Failed to write mask image to '{file_path}'.")
