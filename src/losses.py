from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.losses import SSIMLoss


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def grad3d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    pdx, pdy, pdz = grad3d(pred)
    tdx, tdy, tdz = grad3d(target)
    return (F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy) + F.l1_loss(pdz, tdz)) / 3.0


def ncc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=[2, 3, 4], keepdim=True)
    tgt_c = target - target.mean(dim=[2, 3, 4], keepdim=True)
    numerator = (pred_c * tgt_c).mean(dim=[2, 3, 4])
    denominator = torch.sqrt((pred_c.square().mean(dim=[2, 3, 4]) + eps) * (tgt_c.square().mean(dim=[2, 3, 4]) + eps))
    ncc = numerator / (denominator + eps)
    return 1.0 - ncc.mean()


class CompositeCorrectionLoss(torch.nn.Module):
    def __init__(self, l1_weight: float, ssim_weight: float, grad_weight: float, ncc_weight: float):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.ncc_weight = ncc_weight
        self.ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim(pred, target)
        grad = gradient_loss(pred, target)
        ncc = ncc_loss(pred, target)

        total = self.l1_weight * l1 + self.ssim_weight * ssim + self.grad_weight * grad + self.ncc_weight * ncc
        parts = {
            "l1": float(l1.detach().item()),
            "ssim": float(ssim.detach().item()),
            "grad": float(grad.detach().item()),
            "ncc": float(ncc.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, parts
