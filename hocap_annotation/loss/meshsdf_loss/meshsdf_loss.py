import torch
import torch.nn as nn
import meshsdf_loss_cuda


class MeshSDFLossFunction(torch.autograd.Function):
    """
    Custom autograd function for the MeshSDF loss, allowing forward and backward computations.
    """

    @staticmethod
    def forward(ctx, verts: torch.Tensor, faces: torch.Tensor, points: torch.Tensor):
        """
        Forward pass for the MeshSDFLoss function.
        """
        # Ensure all tensors are on the same device
        device = verts.device
        faces = faces.to(device)
        points = points.to(device)

        # Forward pass using the custom CUDA extension
        outputs = meshsdf_loss_cuda.forward(verts, faces, points)
        loss, dist, assoc, baryc = outputs

        # Save variables needed for backward pass
        ctx.save_for_backward(verts, faces, points, assoc, baryc)

        return loss.to(device), dist.to(device), assoc.to(device)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_dist=None, grad_assoc=None):
        """
        Backward pass for the MeshSDFLoss function.
        """
        # Retrieve saved tensors from the forward pass
        verts, faces, points, assoc, baryc = ctx.saved_tensors
        device = verts.device

        # Ensure gradients are on the correct device
        grad_loss = grad_loss.to(device)

        # Backward pass using the custom CUDA extension
        outputs = meshsdf_loss_cuda.backward(
            grad_loss, verts, faces, points, assoc, baryc
        )
        d_verts = outputs[0]

        return (
            d_verts.to(device),  # Gradient for verts
            None,  # No gradient for faces
            None,  # No gradient for points
        )


class MeshSDFLoss(nn.Module):
    """
    PyTorch module for the MeshSDF loss, wrapping the custom MeshSDFLossFunction.
    """

    def __init__(self, loss_type="mse"):
        """Initializes the MeshSDFLoss module."""
        if loss_type not in ["l2_norm", "mse"]:
            raise ValueError(
                f"Invalid loss type: {loss_type}, must be 'l2_norm' or 'mse'"
            )

        super(MeshSDFLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, verts: torch.Tensor, faces: torch.Tensor, points: torch.Tensor):
        """
        Forward pass for the MeshSDFLoss module.
        """
        # Move inputs to the same device
        device = verts.device
        faces = faces.to(device)
        points = points.to(device)

        loss, dist, assoc = MeshSDFLossFunction.apply(verts, faces, points)

        if self.loss_type == "l2_norm":  # L2 Euclidean distance loss
            loss = torch.mean(dist)
        else:
            loss *= 1e3 / points.size(0)  # Scale to meters

        return loss, dist, assoc
