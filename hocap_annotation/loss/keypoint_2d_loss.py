import torch
import torch.nn as nn


class Keypoint2DLoss(nn.Module):
    def __init__(self, loss_type="l2_norm"):
        super(Keypoint2DLoss, self).__init__()
        if loss_type == "l1":
            self._loss_fn = nn.L1Loss(reduction="sum")
        elif loss_type == "l2_norm":
            self._loss_fn = self._l2_norm_loss
        elif loss_type == "l2":
            self._loss_fn = nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        self.register_buffer("_zero", torch.tensor(0.0, dtype=torch.float32))

    def _l2_norm_loss(self, X, Y):
        return torch.linalg.vector_norm(X - Y, ord=2, dim=-1).sum()

    def forward(self, pred_kpts_2d, gt_kpts_2d, valid_mask=None):
        """
        Args:
            pred_kpts_2d: (B, N, 2), predicted 2D keypoints
            gt_kpts_2d: (B, N, 2), ground truth 2D keypoints
            valid_mask: (B, N), mask indicating valid keypoints
        """
        mask = (
            (pred_kpts_2d[..., 0] >= 0)
            & (pred_kpts_2d[..., 0] < 1)
            & (pred_kpts_2d[..., 1] >= 0)
            & (pred_kpts_2d[..., 1] < 1)
        )

        if valid_mask is not None:
            mask = mask & valid_mask

        num_valid_kpts = torch.sum(mask)

        if num_valid_kpts == 0:
            return self._zero

        loss = self._loss_fn(pred_kpts_2d[mask], gt_kpts_2d[mask]) / num_valid_kpts
        return loss
