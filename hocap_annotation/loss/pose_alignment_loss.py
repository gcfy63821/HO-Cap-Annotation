import torch
import torch.nn as nn


class PoseAlignmentLoss(nn.Module):
    def __init__(self, loss_type="l2_norm"):
        super(PoseAlignmentLoss, self).__init__()

        if loss_type == "l1":
            self._loss = nn.L1Loss(reduction="sum")
        elif loss_type == "l2_norm":
            self._loss_fn = self._l2_norm_loss
        elif self.loss_type == "l2":
            self._loss_fn = nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

    def _l2_norm_loss(self, X, Y):
        return torch.linalg.vector_norm(X - Y, ord=2, dim=-1).sum()

    def forward(self, poses_A, poses_B, subset=None):
        """
        Args:
            poses_A: list of torch.Tensor, each tensor has shape (batch_size, pose_dim)
            poses_B: list of torch.Tensor, each tensor has shape (batch_size, pose_dim)
            subset: list of int, the indices of the selected poses
        """

        if subset is None:
            subset = list(range(len(poses_A)))

        X = torch.stack([poses_A[i] for i in subset], dim=0).view(-1, 3)
        Y = torch.stack([poses_B[i] for i in subset], dim=0).view(-1, 3)

        loss = self._loss_fn(X, Y)
        loss /= X.size(0)
        return loss
