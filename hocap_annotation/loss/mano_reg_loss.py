import torch
import torch.nn as nn


class MANORegLoss(nn.Module):
    def __init__(self):
        super(MANORegLoss, self).__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, poses, subset=None):
        if subset is None:
            subset = list(range(len(poses)))
        X = torch.stack([poses[i][..., 3:48] for i in subset], dim=0)
        num_objs, num_frames, _ = X.shape
        loss = self._mse_loss(X, torch.zeros_like(X))
        loss /= num_objs * num_frames
        return loss
