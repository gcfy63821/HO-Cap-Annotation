from .common_imports import *


def average_quats(quats: np.ndarray) -> np.ndarray:
    """
    Calculate the average quaternion from a set of quaternions.

    Args:
        quats (np.ndarray): An array of quaternions of shape (N, 4), where N is the number of quaternions.

    Returns:
        np.ndarray: The averaged quaternion of shape (4,).
    """
    if not isinstance(quats, np.ndarray) or quats.shape[-1] != 4:
        raise ValueError("Input must be a numpy array of shape (N, 4).")

    rotations = R.from_quat(quats)
    avg_quat = rotations.mean().as_quat().astype(np.float32)
    return avg_quat


def normalize_quats(qs: np.ndarray) -> np.ndarray:
    """
    Normalize quaternions to have unit length.

    Args:
        qs (np.ndarray): Input quaternion, shape (4,) or (N, 4) where each quaternion is (qx, qy, qz, qw).

    Returns:
        np.ndarray: Normalized quaternion(s), same shape as input.
    """
    # Compute the norm of the quaternion
    norms = np.linalg.norm(qs, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Quaternion norms cannot be zero.")
    return qs / norms


def rvt_to_quat(rvt: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to quaternion and translation vector.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) for single or (N, 6) for batch.

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) for single or (N, 7) for batch,
                    in the format [qx, qy, qz, qw, tx, ty, tz].
    """
    # Ensure the input has the correct shape
    if rvt.ndim == 1 and rvt.shape[0] == 6:
        rv = rvt[:3]
        t = rvt[3:]
        q = R.from_rotvec(rv).as_quat()
        return np.concatenate([q, t], dtype=np.float32)

    elif rvt.ndim == 2 and rvt.shape[1] == 6:
        rv = rvt[:, :3]
        t = rvt[:, 3:]
        q = R.from_rotvec(rv).as_quat()  # Batch process
        return np.concatenate([q, t], axis=-1).astype(np.float32)

    else:
        raise ValueError("Input must be of shape (6,) or (N, 6).")


def quat_to_rvt(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion and translation vector to rotation vector and translation vector.

    Args:
        quat (np.ndarray): Quaternion and translation vector. Shape can be (7,) for single input
                           or (N, 7) for batched input.

    Returns:
        np.ndarray: Rotation vector and translation vector. Shape will be (6,) for single input
                    or (N, 6) for batched input.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    # Validate input shape
    if not isinstance(quat, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if quat.ndim == 1 and quat.shape[0] == 7:
        batch_mode = False
    elif quat.ndim == 2 and quat.shape[1] == 7:
        batch_mode = True
    else:
        raise ValueError(
            "Input must have shape (7,) for a single quaternion or (N, 7) for a batch of quaternions."
        )

    # Extract quaternion (q) and translation (t)
    q = quat[..., :4]  # Quaternion (4 elements)
    t = quat[..., 4:]  # Translation (3 elements)

    # Convert quaternion to rotation vector
    r = R.from_quat(q)
    rv = r.as_rotvec()  # Convert to rotation vector (3 elements)

    # Concatenate rotation vector and translation vector
    return np.concatenate([rv, t], axis=-1).astype(np.float32)


def rvt_to_mat(rvt: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to pose matrix.

    Args:
        rvt (np.ndarray): Rotation vector and translation vector, shape (6,) for single or (N, 6) for batch.

    Returns:
        np.ndarray: Pose matrix, shape (4, 4) for single or (N, 4, 4) for batch.
    """
    # Single input case (shape (6,))
    if rvt.ndim == 1 and rvt.shape[0] == 6:
        p = np.eye(4)
        rv = rvt[:3]
        t = rvt[3:]
        r = R.from_rotvec(rv)
        p[:3, :3] = r.as_matrix()
        p[:3, 3] = t
        return p.astype(np.float32)

    # Batched input case (shape (N, 6))
    elif rvt.ndim == 2 and rvt.shape[1] == 6:
        N = rvt.shape[0]
        p = np.tile(np.eye(4), (N, 1, 1))  # Create an identity matrix for each batch
        rv = rvt[:, :3]  # Rotation vectors (N, 3)
        t = rvt[:, 3:]  # Translation vectors (N, 3)
        r = R.from_rotvec(rv)
        p[:, :3, :3] = r.as_matrix()  # Set rotation matrices for each batch
        p[:, :3, 3] = t  # Set translation vectors for each batch
        return p.astype(np.float32)

    else:
        raise ValueError("Input must be of shape (6,) or (N, 6).")


def mat_to_rvt(mat_4x4: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix to rotation vector and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) for single input
                              or (N, 4, 4) for batched input.

    Returns:
        np.ndarray: Rotation vector and translation vector, shape (6,) for single input
                    or (N, 6) for batched input.
    """
    # Single input case (shape (4, 4))
    if mat_4x4.ndim == 2 and mat_4x4.shape == (4, 4):
        r = R.from_matrix(mat_4x4[:3, :3])
        rv = r.as_rotvec()
        t = mat_4x4[:3, 3]
        return np.concatenate([rv, t], dtype=np.float32)

    # Batched input case (shape (N, 4, 4))
    elif mat_4x4.ndim == 3 and mat_4x4.shape[1:] == (4, 4):
        rv = R.from_matrix(mat_4x4[:, :3, :3]).as_rotvec()  # Batch process rotations
        t = mat_4x4[:, :3, 3]  # Batch process translations
        return np.concatenate([rv, t], axis=-1).astype(np.float32)

    else:
        raise ValueError("Input must be of shape (4, 4) or (N, 4, 4).")


def mat_to_quat(mat_4x4: np.ndarray) -> np.ndarray:
    """
    Convert pose matrix to quaternion and translation vector.

    Args:
        mat_4x4 (np.ndarray): Pose matrix, shape (4, 4) for single input or (N, 4, 4) for batched input.

    Returns:
        np.ndarray: Quaternion and translation vector, shape (7,) for single input or (N, 7) for batched input.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    if not isinstance(mat_4x4, np.ndarray) or mat_4x4.shape[-2:] != (4, 4):
        raise ValueError("Input must be a numpy array with shape (4, 4) or (N, 4, 4).")

    if np.all(mat_4x4 == -1):
        if mat_4x4.ndim == 2:  # Single matrix (shape (4, 4))
            return np.full((7,), -1, dtype=np.float32)
        elif mat_4x4.ndim == 3:  # Batch of matrices (shape (N, 4, 4))
            return np.full((mat_4x4.shape[0], 7), -1, dtype=np.float32)

    if mat_4x4.ndim == 2:  # Single matrix (shape (4, 4))
        r = R.from_matrix(mat_4x4[:3, :3])
        q = r.as_quat()  # Quaternion (shape (4,))
        t = mat_4x4[:3, 3]  # Translation (shape (3,))
        return np.concatenate([q, t], dtype=np.float32)

    elif mat_4x4.ndim == 3:  # Batch of matrices (shape (N, 4, 4))
        r = R.from_matrix(mat_4x4[:, :3, :3])  # Handle batch of rotation matrices
        q = r.as_quat()  # Quaternions (shape (N, 4))
        t = mat_4x4[:, :3, 3]  # Translations (shape (N, 3))
        return np.concatenate([q, t], axis=-1).astype(np.float32)  # Shape (N, 7)

    else:
        raise ValueError("Input dimension is not valid. Must be 2D or 3D.")


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion and translation vector to a pose matrix.

    This function supports converting a single quaternion or a batch of quaternions.

    Args:
        quat (np.ndarray): Quaternion and translation vector. Shape can be (7,) for a single quaternion
                           or (N, 7) for a batch of quaternions, where N is the batch size.

    Returns:
        np.ndarray: Pose matrix. Shape will be (4, 4) for a single quaternion or (N, 4, 4) for a batch of quaternions.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    # Validate input shape
    if not isinstance(quat, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if quat.ndim == 1 and quat.shape[0] == 7:
        batch_mode = False
    elif quat.ndim == 2 and quat.shape[1] == 7:
        batch_mode = True
    else:
        raise ValueError(
            "Input must have shape (7,) for a single quaternion or (N, 7) for a batch of quaternions."
        )

    # Extract quaternion (q) and translation (t)
    q = quat[..., :4]  # Quaternion (4 elements)
    t = quat[..., 4:]  # Translation (3 elements)

    # Prepare the pose matrix
    if batch_mode:
        N = quat.shape[0]
        p = np.tile(np.eye(4), (N, 1, 1))  # Create N identity matrices
    else:
        p = np.eye(4)  # Single identity matrix

    # Convert quaternion to rotation matrix and fill in the pose matrix
    r = R.from_quat(q)
    p[..., :3, :3] = r.as_matrix()  # Fill rotation part
    p[..., :3, 3] = t  # Fill translation part

    return p.astype(np.float32)


def quat_distance(
    q1: np.ndarray, q2: np.ndarray, in_degree: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate the shortest angular distance in degrees between paired quaternions.

    Args:
        q1 (np.ndarray): First quaternion(s), shape (4,) or (N, 4).
        q2 (np.ndarray): Second quaternion(s), shape (4,) or (N, 4).

    Returns:
        float or np.ndarray: Angular distance in degrees, scalar if single pair, array if multiple pairs.
    """
    # Validate input shapes
    if q1.ndim not in {1, 2} or q2.ndim not in {1, 2}:
        raise ValueError("q1 and q2 must be 1D or 2D arrays.")
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("Each quaternion must have 4 components (qx, qy, qz, qw).")
    if q1.shape != q2.shape:
        raise ValueError("q1 and q2 must have the same shape.")

    # Normalize quaternions to ensure they are unit quaternions
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    # Compute the dot product between paired quaternions
    dot_product = np.sum(q1 * q2, axis=-1)

    # Clamp the dot product to the range [-1, 1] to handle numerical precision issues
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the shortest angular distance in radians
    angular_distance = 2 * np.arccos(np.abs(dot_product))

    # Convert to degrees if needed
    if in_degree:
        return np.degrees(angular_distance)
    return angular_distance


def trans_distance(t1, t2):
    """Calculate the Euclidean distance between two translation vectors or arrays of translation vectors.

    Args:
        t1 (np.ndarray): First translation vector(s) in shape (3,) or (N, 3), where N is the number of vectors.
        t2 (np.ndarray): Second translation vector(s) in shape (3,) or (N, 3), where N is the number of vectors.

    Returns:
        float or np.ndarray: Euclidean distance. Returns a scalar if inputs are 1D vectors, or an array of distances if inputs are 2D arrays.
    Raises:
        ValueError: If the inputs are not valid translation vectors or if their shapes are incompatible.
    """

    # Ensure both inputs are NumPy arrays
    t1 = np.asarray(t1, dtype=np.float32)
    t2 = np.asarray(t2, dtype=np.float32)

    # Check if the shapes of t1 and t2 are compatible
    if t1.shape != t2.shape:
        raise ValueError(
            f"Shape mismatch: t1.shape {t1.shape} and t2.shape {t2.shape} must be the same."
        )

    # Check for valid shapes: (3,) for a single vector or (N, 3) for multiple vectors
    if t1.shape[-1] != 3:
        raise ValueError("Each translation vector must have 3 components (tx, ty, tz).")

    # Compute Euclidean distance
    return np.linalg.norm(t1 - t2, axis=-1)


def quat_to_rv_tensor(quat):
    """Converts quaternions to rotation vectors (axis-angle format).

    Args:
        quat: A tensor of shape [B, 4] containing the quaternions in (x, y, z, w) format.

    Returns:
        rv: A tensor of shape [B, 3] containing the rotation vectors.
    """
    # Normalize the quaternion to ensure it is a unit quaternion
    quat = quat / quat.norm(dim=-1, keepdim=True)

    # Extract vector (xyz) and scalar (w) parts
    xyz = quat[..., :3]  # Shape [B, 3]
    w = quat[..., 3]  # Shape [B]

    # Calculate the angle: theta = 2 * arccos(w)
    theta = 2 * torch.acos(
        w.clamp(min=-1.0, max=1.0)
    )  # Clamp to avoid numerical issues

    # Calculate sin(theta / 2) to avoid division by zero when theta is very small
    sin_theta_half = torch.sqrt(1 - w**2).clamp(
        min=1e-8
    )  # Use a small min value to prevent division issues

    # Calculate the rotation axis and handle the case where sin(theta/2) is very small
    axis = xyz / sin_theta_half.unsqueeze(-1)

    # Zero out the axis when sin(theta/2) is effectively zero (angle ~ 0)
    axis = torch.where(sin_theta_half > 1e-6, axis, torch.zeros_like(axis))

    # Calculate the rotation vector as axis * angle
    rv = axis * theta.unsqueeze(-1)

    return rv


def quat_to_rot_mat_tensor(quat):
    """Converts quaternions to rotation matrices.

    Args:
        quat: A tensor of shape [B, 4] containing quaternions in (x, y, z, w) format.

    Returns:
        rot_matrix: A tensor of shape [B, 3, 3] containing rotation matrices.
    """
    # Normalize the quaternion
    quat = quat / quat.norm(dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(dim=-1)

    # Compute rotation matrix elements
    rot_matrix = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    return rot_matrix


def rv_to_rot_mat_tensor(rv):
    """Converts rotation vectors (axis-angle format) to rotation matrices.

    Args:
        rv: A tensor of shape [B, 3] containing rotation vectors.

    Returns:
        rot_matrix: A tensor of shape [B, 3, 3] containing rotation matrices.
    """
    theta = torch.norm(rv, dim=-1, keepdim=True)
    axis = torch.where(theta > 1e-6, rv / theta, torch.zeros_like(rv))
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Rodrigues' rotation formula
    K = torch.zeros((rv.shape[0], 3, 3), device=rv.device)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]

    rot_matrix = (
        cos_theta * torch.eye(3, device=rv.device)
        + sin_theta * K
        + (1 - cos_theta) * torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(1))
    )
    return rot_matrix


def rot_mat_to_quat_tensor(rot_matrix):
    """Converts rotation matrices to quaternions.

    Args:
        rot_matrix: A tensor of shape [B, 3, 3] containing rotation matrices.

    Returns:
        quat: A tensor of shape [B, 4] containing quaternions in (x, y, z, w) format.
    """
    trace = rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
    quat = torch.zeros((rot_matrix.size(0), 4), device=rot_matrix.device)

    # Compute quaternion values based on trace
    cond = trace > 0
    quat[cond, 3] = 0.5 * torch.sqrt(1.0 + trace[cond])
    quat[cond, 0] = (rot_matrix[cond, 2, 1] - rot_matrix[cond, 1, 2]) / (
        4 * quat[cond, 3]
    )
    quat[cond, 1] = (rot_matrix[cond, 0, 2] - rot_matrix[cond, 2, 0]) / (
        4 * quat[cond, 3]
    )
    quat[cond, 2] = (rot_matrix[cond, 1, 0] - rot_matrix[cond, 0, 1]) / (
        4 * quat[cond, 3]
    )

    # For cases with trace <= 0, use alternative calculations
    max_diag = torch.argmax(
        torch.stack(
            [rot_matrix[:, 0, 0], rot_matrix[:, 1, 1], rot_matrix[:, 2, 2]], dim=1
        ),
        dim=1,
    )
    for i, md in enumerate(max_diag):
        if not cond[i]:
            if md == 0:
                s = 2 * torch.sqrt(
                    1.0
                    + rot_matrix[i, 0, 0]
                    - rot_matrix[i, 1, 1]
                    - rot_matrix[i, 2, 2]
                )
                quat[i, 0] = 0.25 * s
                quat[i, 1] = (rot_matrix[i, 0, 1] + rot_matrix[i, 1, 0]) / s
                quat[i, 2] = (rot_matrix[i, 0, 2] + rot_matrix[i, 2, 0]) / s
                quat[i, 3] = (rot_matrix[i, 2, 1] - rot_matrix[i, 1, 2]) / s
            elif md == 1:
                s = 2 * torch.sqrt(
                    1.0
                    + rot_matrix[i, 1, 1]
                    - rot_matrix[i, 0, 0]
                    - rot_matrix[i, 2, 2]
                )
                quat[i, 0] = (rot_matrix[i, 0, 1] + rot_matrix[i, 1, 0]) / s
                quat[i, 1] = 0.25 * s
                quat[i, 2] = (rot_matrix[i, 1, 2] + rot_matrix[i, 2, 1]) / s
                quat[i, 3] = (rot_matrix[i, 0, 2] - rot_matrix[i, 2, 0]) / s
            else:
                s = 2 * torch.sqrt(
                    1.0
                    + rot_matrix[i, 2, 2]
                    - rot_matrix[i, 0, 0]
                    - rot_matrix[i, 1, 1]
                )
                quat[i, 0] = (rot_matrix[i, 0, 2] + rot_matrix[i, 2, 0]) / s
                quat[i, 1] = (rot_matrix[i, 1, 2] + rot_matrix[i, 2, 1]) / s
                quat[i, 2] = 0.25 * s
                quat[i, 3] = (rot_matrix[i, 1, 0] - rot_matrix[i, 0, 1]) / s

    return quat


def rot_mat_to_rv_tensor(rot_matrix):
    """Converts rotation matrices to rotation vectors (axis-angle format).

    Args:
        rot_matrix: A tensor of shape [B, 3, 3] containing rotation matrices.

    Returns:
        rv: A tensor of shape [B, 3] containing rotation vectors.
    """
    trace = rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
    theta = torch.acos((trace - 1) / 2).clamp(min=1e-6)
    sin_theta = torch.sin(theta)

    # Calculate rotation axis using off-diagonal elements
    rx = (rot_matrix[:, 2, 1] - rot_matrix[:, 1, 2]) / (2 * sin_theta)
    ry = (rot_matrix[:, 0, 2] - rot_matrix[:, 2, 0]) / (2 * sin_theta)
    rz = (rot_matrix[:, 1, 0] - rot_matrix[:, 0, 1]) / (2 * sin_theta)

    # Combine axis and angle to get the rotation vector
    rv = theta.unsqueeze(-1) * torch.stack((rx, ry, rz), dim=-1)

    return rv


def average_quats_tensor(quats):
    """
    Average a batch of quaternions.

    Args:
        quats (torch.Tensor): A tensor of shape [B, 4] with quaternions in (x, y, z, w) format.

    Returns:
        torch.Tensor: A tensor of shape [4] representing the average quaternion.
    """
    # Convert to rotation matrices
    rot_matrices = quat_to_rot_mat_tensor(quats)

    # Compute average rotation matrix
    avg_rot_matrix = rot_matrices.mean(dim=0)

    # Convert the average rotation matrix back to a quaternion
    avg_quat = rot_to_quat_tensor(avg_rot_matrix.unsqueeze(0)).squeeze(0)

    # Normalize the resulting average quaternion
    avg_quat = avg_quat / avg_quat.norm()

    return avg_quat


def average_rot_mats_tensor(rot_matrices):
    """
    Average a batch of rotation matrices.

    Args:
        rot_matrices (torch.Tensor): A tensor of shape [B, 3, 3].

    Returns:
        torch.Tensor: A tensor of shape [3, 3] representing the average rotation matrix.
    """
    # Compute the mean rotation matrix
    avg_rot_matrix = rot_matrices.mean(dim=0)

    # Convert to quaternion and back to ensure it's a valid rotation
    avg_quat = rot_to_quat_tensor(avg_rot_matrix.unsqueeze(0)).squeeze(0)
    avg_rot_matrix = quat_to_rot_tensor(avg_quat.unsqueeze(0)).squeeze(0)

    return avg_rot_matrix


def average_rvs_tensor(rvs):
    """
    Average a batch of rotation vectors.

    Args:
        rvs (torch.Tensor): A tensor of shape [B, 3] with rotation vectors.

    Returns:
        torch.Tensor: A tensor of shape [3] representing the average rotation vector.
    """
    # Convert rotation vectors to quaternions
    quats = rv_to_quat_tensor(rvs)

    # Average the quaternions
    avg_quat = average_quats_tensor(quats)

    # Convert back to a rotation vector
    avg_rv = quat_to_rv_tensor(avg_quat.unsqueeze(0)).squeeze(0)

    return avg_rv


def average_trans_tensor(translations):
    """
    Average a batch of translations.

    Args:
        translations (torch.Tensor): A tensor of shape [B, 3] with translations.

    Returns:
        torch.Tensor: A tensor of shape [3] representing the average translation.
    """
    return translations.mean(dim=0)


def rv_to_quat_tensor(rvs):
    """
    Convert rotation vectors to quaternions.

    Args:
        rvs (torch.Tensor): Rotation vectors of shape [B, 3] or [3,].

    Returns:
        torch.Tensor: Quaternions of shape [B, 4] or [4,].
    """
    # Compute the angle (magnitude) of the rotation vector
    angles = rvs.norm(dim=-1, keepdim=True)

    # Normalize rotation vectors to get the axis
    axes = torch.where(angles > 0, rvs / angles, rvs)

    # Compute the quaternion components
    half_angles = angles / 2.0
    quats = torch.cat([axes * torch.sin(half_angles), torch.cos(half_angles)], dim=-1)

    return quats


def quat_distance_tensor(quats_A, quats_B):
    """
    Calculate the shortest angular distance between batched quaternions.

    Args:
        quats_A (torch.Tensor): A tensor of shape [B, 4] containing quaternions in (x, y, z, w) format.
        quats_B (torch.Tensor): A tensor of shape [B, 4] containing quaternions in (x, y, z, w) format.

    Returns:
        torch.Tensor: A tensor of shape [B] containing the rotation distances in radians.
    """
    # Ensure quaternions are normalized
    quats_A = quats_A / quats_A.norm(dim=-1, keepdim=True)
    quats_B = quats_B / quats_B.norm(dim=-1, keepdim=True)

    # Compute dot product and take absolute value
    dot_product = torch.sum(quats_A * quats_B, dim=-1).abs()

    # Clamp dot product to [0, 1] to avoid numerical issues with acos
    dot_product = dot_product.clamp(max=1.0)

    # Calculate the angular distance in radians
    theta = 2 * torch.acos(dot_product)

    return theta


def rv_distance_tensor(rvs_A, rvs_B):
    """
    Calculate the shortest angular distance between batched rotation vectors.

    Args:
        rvs_A (torch.Tensor): A tensor of shape [B, 3] containing rotation vectors (rx, ry, rz).
        rvs_B (torch.Tensor): A tensor of shape [B, 3] containing rotation vectors (rx, ry, rz).

    Returns:
        torch.Tensor: A tensor of shape [B] containing the rotation distances in radians.
    """
    # Ensure inputs have the correct shape
    if rvs_A.shape[-1] != 3 or rvs_B.shape[-1] != 3:
        raise ValueError("Each rotation vector must have 3 components (rx, ry, rz).")
    if rvs_A.shape != rvs_B.shape:
        raise ValueError("rvs_A and rvs_B must have the same shape.")

    # Convert rotation vectors to quaternions using the torch API
    quats_A = rvs_to_quats(rvs_A)
    quats_B = rvs_to_quats(rvs_B)

    # Calculate the angular distance between the quaternions
    return quat_distance_tensor(quats_A, quats_B)


def rv_distance_tensor(rvs_A, rvs_B, in_degree=False, to_quat=False):
    """
    Calculate the rotation distance between batched rotation vectors.

    Args:
        rvs_A (torch.Tensor): A tensor of shape [B, 3] containing rotation vectors.
        rvs_B (torch.Tensor): A tensor of shape [B, 3] containing rotation vectors.
        in_degree (bool): If True, return the distance in degrees; otherwise, in radians.
        to_quat (bool): If True, convert rotation vectors to quaternions for more accurate angular distance.
                        Because the Euclidean distance between rotation vectors does not account for the fact that
                        two vectors with small differences may actually represent a large angular difference in 3D space.

    Returns:
        torch.Tensor: A tensor of shape [B] containing the rotation distances.
    """
    if to_quat:
        # Convert rotation vectors to quaternions
        quats_A = rv_to_quat_tensor(rvs_A)
        quats_B = rv_to_quat_tensor(rvs_B)

        # Compute the quaternion-based rotation distance
        dot_product = torch.sum(quats_A * quats_B, dim=-1).abs()
        theta = 2 * torch.acos(dot_product.clamp(max=1.0))
    else:
        # Direct Euclidean distance between rotation vectors
        theta = torch.norm(rvs_A - rvs_B, dim=-1)

    # Convert to degrees if needed
    if in_degree:
        theta = torch.rad2deg(theta)

    return theta


def rot_mat_distance_tensor(mat_A, mat_B, in_degree=False):
    """
    Calculate the rotation distance between batched rotation matrices.

    Args:
        mat_A (torch.Tensor): A tensor of shape [B, 3, 3] containing rotation matrices.
        mat_B (torch.Tensor): A tensor of shape [B, 3, 3] containing rotation matrices.
        in_degree (bool): If True, return the distance in degrees; otherwise, in radians.

    Returns:
        torch.Tensor: A tensor of shape [B] containing the rotation distances.
    """
    # Compute the relative rotation matrix
    relative_rot = torch.bmm(mat_A.transpose(1, 2), mat_B)

    # Compute the trace of the relative rotation matrix
    trace = relative_rot[:, 0, 0] + relative_rot[:, 1, 1] + relative_rot[:, 2, 2]

    # Calculate the angle
    theta = torch.acos(
        ((trace - 1) / 2).clamp(-1.0, 1.0)
    )  # Clamp to avoid numerical errors

    # Convert to degrees if needed
    if in_degree:
        theta = torch.rad2deg(theta)

    return theta


def trans_distance_tensor(trans_A, trans_B):
    """
    Calculate the Euclidean distance between batched translations.

    Args:
        trans_A (torch.Tensor): A tensor of shape [B, 3] containing translations.
        trans_B (torch.Tensor): A tensor of shape [B, 3] containing translations.

    Returns:
        torch.Tensor: A tensor of shape [B] containing the translation distances.
    """
    # Compute the Euclidean distance
    distance = torch.norm(trans_A - trans_B, dim=-1)

    return distance


def angular_difference(q1: np.ndarray, q2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate the angular difference in degrees between two quaternions or arrays of quaternions.

    Args:
        q1 (np.ndarray): First quaternion(s) in [qx, qy, qz, qw] or [N, qx, qy, qz, qw] format.
        q2 (np.ndarray): Second quaternion(s) in [qx, qy, qz, qw] or [N, qx, qy, qz, qw] format.

    Returns:
        float or np.ndarray: Angular difference in degrees, scalar if single pair or array if multiple pairs.
    """
    dim = q1.ndim
    if dim == 1:
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
    else:
        q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    q1 = R.from_quat(q1)
    q2 = R.from_quat(q2)
    delta_q = q1.inv() * q2
    delta_q_quat = delta_q.as_quat()

    if dim == 1:
        if delta_q_quat[3] < 0:
            delta_q_quat = -delta_q_quat
    else:
        negative_indices = delta_q_quat[:, 3] < 0
        delta_q_quat[negative_indices] = -delta_q_quat[negative_indices]

    if dim == 1:
        angular_diff = 2 * np.arccos(np.clip(delta_q_quat[3], -1.0, 1.0))
    else:
        angular_diff = 2 * np.arccos(np.clip(delta_q_quat[:, 3], -1.0, 1.0))

    return np.degrees(angular_diff)


def evaluate_and_fix_poses(
    poses: np.ndarray,
    window_size: int = 3,
    rot_thresh: float = 1.0,
    trans_thresh: float = 0.01,
    seperate_rot_trans: bool = True,
    use_mean_pose: bool = False,
) -> np.ndarray:
    """
    Evaluate and fix poses.

    Args:
        poses (np.ndarray): Poses to be evaluated and fixed, shape (num_frames, 7) where each row is (qx, qy, qz, qw, x, y, z).
        window_size (int): Window size for smoothing.
        rot_thresh (float): Rotational threshold in degrees.
        trans_thresh (float): Translational threshold.
        seperate_rot_trans (bool): Flag to separate rotation and translation corrections.
        use_mean_pose (bool): Flag to use the mean pose for corrections.

    Returns:
        np.ndarray: Fixed poses.
    """
    num_poses, pose_dim = poses.shape
    fixed_poses = np.copy(poses)

    for i in range(num_poses):
        start = max(0, i - window_size)
        end = min(num_poses, i + window_size + 1)
        current_quat = poses[i, :4]
        current_trans = poses[i, 4:]
        is_trans_static = True
        is_rot_static = True

        for j in range(start, end):
            if j == i:
                continue

            quat = poses[j, :4]
            trans = poses[j, 4:]
            rot_diff = quat_distance(current_quat, quat, in_degree=True)
            # rot_diff = angular_difference(current_quat, quat)
            trans_diff = np.linalg.norm(current_trans - trans)

            if rot_diff > rot_thresh:
                is_rot_static = False
            if trans_diff > trans_thresh:
                is_trans_static = False
            if not is_rot_static and not is_trans_static:
                break

        if seperate_rot_trans:
            if is_rot_static and i > 0:
                fixed_poses[i, :4] = (
                    np.mean(fixed_poses[start:end, :4], axis=0)
                    if use_mean_pose
                    else fixed_poses[i - 1, :4]
                )
            if is_trans_static and i > 0:
                fixed_poses[i, 4:] = (
                    np.mean(fixed_poses[start:end, 4:], axis=0)
                    if use_mean_pose
                    else fixed_poses[i - 1, 4:]
                )
        else:
            if is_rot_static and is_trans_static and i > 0:
                fixed_poses[i] = (
                    np.mean(fixed_poses[start:end], axis=0)
                    if use_mean_pose
                    else fixed_poses[i - 1]
                )

    return fixed_poses


def fix_quaternion(
    curr_quat: Union[np.ndarray, list], prev_quat: Union[np.ndarray, list]
) -> np.ndarray:
    """
    Fix the current quaternion based on the previous quaternion to avoid flipping.

    Args:
        curr_quat (np.ndarray | list): Current quaternion as a list or numpy array.
        prev_quat (np.ndarray | list): Previous quaternion as a list or numpy array.

    Returns:
        np.ndarray: Adjusted current quaternion to avoid flipping.
    """

    # Normalize the current and previous quaternions
    curr_q_norm = np.linalg.norm(curr_quat)
    prev_q_norm = np.linalg.norm(prev_quat)

    if curr_q_norm == 0 or prev_q_norm == 0:
        raise ValueError("Quaternion norm cannot be zero.")

    curr_q = np.array(curr_quat) / curr_q_norm
    prev_q = np.array(prev_quat) / prev_q_norm

    # Convert normalized quaternions to Rotation objects
    curr_rotation = R.from_quat(curr_q)
    prev_rotation = R.from_quat(prev_q)

    # Calculate the relative rotation from curr_rotation to prev_rotation
    delta_rotation = curr_rotation.inv() * prev_rotation

    # Extract the quaternion array from the relative rotation
    delta_quat = delta_rotation.as_quat()

    # Ensure the current quaternion is in the same hemisphere as the previous quaternion
    if delta_quat[3] < 0:
        adjusted_curr_q = -curr_rotation.as_quat()
    else:
        adjusted_curr_q = curr_rotation.as_quat()

    return adjusted_curr_q
