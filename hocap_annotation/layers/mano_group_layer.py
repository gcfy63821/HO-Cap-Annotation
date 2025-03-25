import numpy as np
import torch
from torch.nn import Module, ModuleList
from .mano_layer import MANOLayer


class MANOGroupLayer(Module):
    """
    A wrapper layer to manage multiple MANOLayers for handling multiple hand objects.
    """

    def __init__(self, sides: list[str], betas: list[np.ndarray]):
        """
        Initialize the MANOGroupLayer.

        Args:
            sides (list[str]): A list of sides for the hands ('right' or 'left').
            betas (list[np.ndarray]): A list of numpy arrays, each of shape [10], containing shape parameters (betas).

        Raises:
            ValueError: If any side in 'sides' is not 'right' or 'left'.
        """
        super(MANOGroupLayer, self).__init__()

        # Validate sides input
        if not all(side in ["right", "left"] for side in sides):
            raise ValueError("All entries in 'sides' must be either 'right' or 'left'.")

        self._sides = sides
        self._betas = betas
        self._num_obj = len(self._sides)

        # Create a list of MANOLayers, one for each hand object
        self._layers = ModuleList(
            [MANOLayer(s, b) for s, b in zip(self._sides, self._betas)]
        )

        # Concatenate faces from each MANOLayer and register as a buffer
        f = torch.cat([self._layers[i].f + 778 * i for i in range(self._num_obj)])
        self.register_buffer("f", f)

        # Concatenate root translations and register as a buffer
        r = torch.cat([l.root_trans for l in self._layers])
        self.register_buffer("root_trans", r)

    def forward(
        self, p: torch.Tensor, inds: list[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MANOGroupLayer.

        Args:
            p (torch.Tensor): A tensor of shape [B, D] containing the pose vectors.
            inds (list[int], optional): A list of sub-layer indices. If None, all layers are used.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                v: A tensor of shape [B, N, 3] containing the vertices for each hand.
                j: A tensor of shape [B, J, 3] containing the joints for each hand.
        """
        if inds is None:
            inds = range(self._num_obj)

        # Initialize empty tensors for vertices and joints
        v = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]
        j = [torch.zeros((p.size(0), 0, 3), dtype=torch.float32, device=self.f.device)]

        # Extract pose and translation from input pose vectors
        p, t = self.pose2pt(p)

        # Pass pose and translation to each selected MANOLayer
        for i in inds:
            verts, joints = self._layers[i](p[:, i], t[:, i])
            v.append(verts)
            j.append(joints)

        # Concatenate results
        v = torch.cat(v, dim=1)
        j = torch.cat(j, dim=1)
        return v, j

    def pose2pt(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pose and translation vectors from the input pose tensor.

        Args:
            pose (torch.Tensor): A tensor of shape [B, D] containing the pose vectors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                p: A tensor of shape [B, O, 48] containing the pose parameters.
                t: A tensor of shape [B, O, 3] containing the translation parameters.
        """
        p = torch.stack(
            [pose[:, 51 * i : 51 * i + 48] for i in range(self._num_obj)], dim=1
        )
        t = torch.stack(
            [pose[:, 51 * i + 48 : 51 * i + 51] for i in range(self._num_obj)], dim=1
        )
        return p, t

    def get_f_from_inds(self, inds: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get faces from sub-layer indices.

        Args:
            inds (list[int]): A list of sub-layer indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                f: A tensor of shape [F, 3] containing the faces.
                m: A tensor of shape [F] containing the face-to-index mapping.
        """
        f = [torch.zeros((0, 3), dtype=self.f.dtype, device=self.f.device)]
        m = [torch.zeros((0,), dtype=torch.int64, device=self.f.device)]

        for i, x in enumerate(inds):
            f.append(self._layers[x].f + 778 * i)
            m.append(
                x
                * torch.ones(
                    self._layers[x].f.size(0), dtype=torch.int64, device=self.f.device
                )
            )

        f = torch.cat(f)
        m = torch.cat(m)
        return f, m

    @property
    def num_obj(self) -> int:
        """Return the number of objects (hand layers)."""
        return self._num_obj
