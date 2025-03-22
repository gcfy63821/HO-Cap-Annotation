import torch
import torch.nn as nn


class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type="l2_norm"):
        super(Keypoint3DLoss, self).__init__()

        if loss_type == "l1":
            self._loss_fn = nn.L1Loss(reduction="sum")
        elif loss_type == "l2_norm":
            self._loss_fn = self._l2_norm_loss
        elif loss_type == "mse":
            self._loss_fn = nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        self.register_buffer("_zero", torch.tensor(0.0, dtype=torch.float32))

    def _l2_norm_loss(self, X, Y):
        return torch.linalg.vector_norm(X - Y, ord=2, dim=-1).sum()

    def forward(self, pred_kpts_3d, gt_kpts_3d, valid_indices=None):
        """
        Args:
            pred_kpts_3d: (B, N, 3), predicted 3D keypoints
            gt_kpts_3d: (B, N, 3), ground truth 3D keypoints
            valid_mask: (B, N), mask indicating valid keypoints
        """
        X = (
            pred_kpts_3d[:, valid_indices, :]
            if valid_indices is not None
            else pred_kpts_3d
        )
        Y = gt_kpts_3d[:, valid_indices, :] if valid_indices is not None else gt_kpts_3d

        X = X.view(-1, 3)
        Y = Y.view(-1, 3)

        loss = self._loss_fn(X, Y) / X.size(0)
        return loss
