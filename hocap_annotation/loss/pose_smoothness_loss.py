import torch
import torch.nn as nn


class PoseSmoothnessLoss(nn.Module):
    def __init__(self, win_size=1, w_vel_r=1.0, w_vel_t=1.0, w_acc_r=1.0, w_acc_t=1.0):
        super(PoseSmoothnessLoss, self).__init__()
        self._win_size = win_size
        self._w_vel_r = w_vel_r
        self._w_vel_t = w_vel_t
        self._w_acc_r = w_acc_r
        self._w_acc_t = w_acc_t
        self.register_buffer("_zero", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, poses, subset=None):
        """
        Args:
            poses: list of torch.Tensor, each tensor has shape (batch_size, num_objs, pose_dim)
            subset: list of int, the indices of the selected poses
        """

        if subset is None:
            subset = list(range(len(poses)))

        P = torch.stack([poses[i] for i in subset], dim=1)

        loss_vel_r = self._zero.clone()
        loss_vel_t = self._zero.clone()
        loss_acc_r = self._zero.clone()
        loss_acc_t = self._zero.clone()

        for j in range(1, self._win_size + 1):
            vel_forward = P[j:] - P[:-j]
            vel_backward = P[:-j] - P[j:]
            acc_forward = vel_forward[1:] - vel_forward[:-1]
            acc_backward = vel_backward[1:] - vel_backward[:-1]
            vel_r_diff = torch.cat(
                [vel_forward[..., :-3], vel_backward[..., :-3]], dim=0
            ).view(-1, 3)
            vel_t_diff = torch.cat(
                [vel_forward[..., 3:], vel_backward[..., 3:]], dim=0
            ).view(-1, 3)
            acc_r_diff = torch.cat(
                [acc_forward[..., :-3], acc_backward[..., :-3]], dim=0
            ).view(-1, 3)
            acc_t_diff = torch.cat(
                [acc_forward[..., 3:], acc_backward[..., 3:]], dim=0
            ).view(-1, 3)
            loss_vel_r += (
                self._w_vel_r
                * torch.linalg.vector_norm(vel_r_diff, ord=2, dim=-1).mean()
            )
            loss_vel_t += (
                self._w_vel_t
                * torch.linalg.vector_norm(vel_t_diff, ord=2, dim=-1).mean()
            )
            loss_acc_r += (
                self._w_acc_r
                * torch.linalg.vector_norm(acc_r_diff, ord=2, dim=-1).mean()
            )
            loss_acc_t += (
                self._w_acc_t
                * torch.linalg.vector_norm(acc_t_diff, ord=2, dim=-1).mean()
            )

        total_loss = loss_vel_r + loss_vel_t + loss_acc_r + loss_acc_t
        total_loss /= self._win_size
        return total_loss
