import numpy as np
import torch
from torch.nn import Module
from manopth.manolayer import ManoLayer
from hocap_annotation.utils import  CFG


class MANOLayer(Module):
    """Wrapper layer for manopth ManoLayer."""

    def __init__(self, side: str, betas: np.ndarray):
        """
        Constructor for MANOLayer.

        Args:
            side (str): MANO hand type. 'right' or 'left'.
            betas (np.ndarray): A numpy array of shape [10] containing the betas.
        """
        super(MANOLayer, self).__init__()

        self._side = side
        self._betas = betas
        self._mano_layer = ManoLayer(
            side=side,
            mano_root=CFG.mano.model_path,
            flat_hand_mean=CFG.mano.flat_hand_mean,
            ncomps=CFG.mano.num_pca_comps,
            use_pca=CFG.mano.use_pca,
        )

        # Register buffer for betas
        b = torch.from_numpy(betas).unsqueeze(0).float()
        self.register_buffer("b", b)

        # Register buffer for faces
        self.register_buffer("f", self._mano_layer.th_faces)

        # Register buffer for root translation
        v = (
            torch.matmul(self._mano_layer.th_shapedirs, self.b.transpose(0, 1)).permute(
                2, 0, 1
            )
            + self._mano_layer.th_v_template
        )
        r = torch.matmul(self._mano_layer.th_J_regressor[0], v)
        self.register_buffer("root_trans", r)

    def forward(
        self, p: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward function.

        Args:
            p (torch.Tensor): A tensor of shape [B, 48] containing the pose.
            t (torch.Tensor): A tensor of shape [B, 3] containing the translation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                v: A tensor of shape [B, 778, 3] containing the vertices.
                j: A tensor of shape [B, 21, 3] containing the joints.
        """
        v, j = self._mano_layer(p, self.b.expand(p.size(0), -1), t)

        # Convert to meters.
        v /= 1000.0
        j /= 1000.0
        return v, j

    @property
    def th_hands_mean(self) -> torch.Tensor:
        """Return the hand mean tensor."""
        return self._mano_layer.th_hands_mean

    @property
    def th_selected_comps(self) -> torch.Tensor:
        """Return the selected components tensor."""
        return self._mano_layer.th_selected_comps

    @property
    def th_v_template(self) -> torch.Tensor:
        """Return the vertex template tensor."""
        return self._mano_layer.th_v_template

    @property
    def side(self) -> str:
        """Return the side of the hand."""
        return self._side

    @property
    def num_verts(self) -> int:
        """Return the number of vertices."""
        return 778


if __name__ == "__main__":
    # Test MANOLayer
    side = "left"
    betas = np.zeros(10)
    mano_layer = MANOLayer(side, betas)

    p = torch.zeros(1, 48)
    t = torch.zeros(1, 3)
    v, j = mano_layer(p, t)
    faces = mano_layer.f.cpu().numpy()
    verts = v.cpu().numpy()
    joints = j.cpu().numpy()

    print(f"Vertices: {verts.shape}")
    print(f"Joints: {joints.shape}")
    print(f"Faces: {faces.shape}")
    print(f"Side: {mano_layer.side}")

    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    # save mesh
    o3d.io.write_triangle_mesh(f"hand_{side}.obj", mesh)
    print("Mesh saved.")
