import torch
import torch.nn as nn
import meshsdf_loss_cuda


class MeshSDFLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, verts, faces, points):
        outputs = meshsdf_loss_cuda.forward(verts, faces, points)
        loss = outputs[0]
        dist = outputs[1]
        assoc = outputs[2]
        baryc = outputs[3]
        variables = [verts, faces, points, assoc, baryc]
        ctx.save_for_backward(*variables)
        return loss, dist, assoc

    @staticmethod
    def backward(ctx, grad_loss, grad_dist, grad_assoc):
        outputs = meshsdf_loss_cuda.backward(grad_loss, *ctx.saved_variables)
        d_verts = outputs[0]
        return d_verts, None, None


class MeshSDFLoss(nn.Module):
    def __init__(self, loss_type="l2"):
        super(MeshSDFLoss, self).__init__()

        if loss_type not in ["l2_norm", "l2"]:
            raise ValueError(f"Invalid loss type: {loss_type}")

        self._loss_type = loss_type
        self.register_buffer("_zero", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, verts: torch.Tensor, faces: torch.Tensor, points: torch.Tensor):
        _, dist, assoc = MeshSDFLossFunction.apply(verts, faces, points)
        if dist.size(0) == 0:
            loss = self._zero
        elif self._loss_type == "l2_norm":
            loss = torch.sqrt(dist).sum() / dist.size(0)
        elif self._loss_type == "l2":
            loss = torch.sum(dist) / dist.size(0)  # MSE loss

        return loss, dist, assoc
