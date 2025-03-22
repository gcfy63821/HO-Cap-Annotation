import numpy as np
import torch
from torch.nn import Module


class ObjectLayer(Module):
    """
    A layer representing a 3D object with vertices, faces, and normals, allowing for transformation
    through rotation and translation.
    """

    def __init__(self, verts: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        """
        Initializes the ObjectLayer.

        Args:
            verts (np.ndarray): A numpy array of shape [N, 3] containing the vertices.
            faces (np.ndarray): A numpy array of shape [F, 3] containing the faces (triangular surface elements).
            normals (np.ndarray): A numpy array of shape [N, 3] containing the normals for each vertex.
        """
        super().__init__()

        self._num_verts = verts.shape[0]

        # Convert numpy arrays to torch tensors
        v = torch.from_numpy(verts.astype(np.float32).T)  # Shape [3, N]
        n = torch.from_numpy(normals.astype(np.float32).T)  # Shape [3, N]
        f = torch.from_numpy(faces.astype(np.int64).reshape((-1, 3)))  # Shape [F, 3]

        # Register buffers for vertices, normals, and faces
        self.register_buffer("v", v)
        self.register_buffer("n", n)
        self.register_buffer("f", f)

    def forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to apply rotation and translation to the vertices and normals.

        Args:
            r (torch.Tensor): A tensor of shape [B, 3] containing the rotation vectors (axis-angle format).
            t (torch.Tensor): A tensor of shape [B, 3] containing the translation vectors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                v: A tensor of shape [B, N, 3] containing the transformed vertices.
                n: A tensor of shape [B, N, 3] containing the transformed normals.
        """
        R = self.rv2dcm(r)
        # Apply rotation and translation to the vertices
        v = torch.matmul(R, self.v).permute(0, 2, 1) + t.unsqueeze(1)
        # Apply rotation to the normals (without translation)
        n = torch.matmul(R, self.n).permute(0, 2, 1)
        return v, n

    def rv2dcm(self, rv: torch.Tensor) -> torch.Tensor:
        """
        Converts rotation vectors (axis-angle) to direction cosine matrices (DCMs).

        Args:
            rv (torch.Tensor): A tensor of shape [B, 3] containing the rotation vectors.

        Returns:
            torch.Tensor: A tensor of shape [B, 3, 3] containing the direction cosine matrices (DCMs).
        """
        # Compute the magnitude (angle) of the rotation vectors
        angle = (
            torch.norm(rv, p=2, dim=1, keepdim=True) + 1e-8
        )  # Avoid division by zero

        # Normalize the rotation vectors to get the axis of rotation
        axis = rv / angle

        # Compute sine and cosine of the angles
        s = torch.sin(angle).unsqueeze(2)
        c = torch.cos(angle).unsqueeze(2)

        # Identity matrix and zero tensor
        I = torch.eye(3, device=rv.device).expand(rv.size(0), -1, -1)
        z = torch.zeros_like(axis[:, 0])

        # Skew-symmetric cross-product matrix for the axis
        K = torch.stack(
            (
                torch.stack((z, -axis[:, 2], axis[:, 1]), dim=1),
                torch.stack((axis[:, 2], z, -axis[:, 0]), dim=1),
                torch.stack((-axis[:, 1], axis[:, 0], z), dim=1),
            ),
            dim=1,
        )

        # Compute the direction cosine matrix using Rodrigues' rotation formula
        dcm = I + s * K + (1 - c) * torch.bmm(K, K)

        return dcm

    @property
    def num_verts(self) -> int:
        """
        Return the number of vertices in the object.

        Returns:
            int: The number of vertices.
        """
        return self._num_verts
