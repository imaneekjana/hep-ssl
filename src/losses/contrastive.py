"""Two-view cosine InfoNCE and trace-normalized anisotropic InfoNCE."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _core_dtype(*tensors):
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _normalized_views(z1, z2):
    if z1.ndim != 2 or z1.shape != z2.shape or z1.shape[0] < 2:
        raise ValueError("InfoNCE needs two equally sized [B,d] views with B >= 2.")
    if z1.device != z2.device or not z1.is_floating_point() or not z2.is_floating_point():
        raise ValueError("Views must be floating tensors on the same device.")
    z = torch.cat((z1.to(_core_dtype(z1, z2)), z2.to(_core_dtype(z1, z2))), dim=0)
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite contrastive projections.")
    return F.normalize(z, p=2, dim=-1, eps=1e-12)


def positive_indices(batch_size, device=None):
    """Ordering is all view1 followed by all view2; there is one positive."""
    return (torch.arange(2 * batch_size, device=device) + batch_size) % (2 * batch_size)


def _masked_cross_entropy(scores, batch_size):
    self_mask = torch.eye(2 * batch_size, device=scores.device, dtype=torch.bool)
    return F.cross_entropy(scores.masked_fill(self_mask, -torch.inf),
                           positive_indices(batch_size, scores.device))


class CosineInfoNCE(nn.Module):
    def __init__(self, tau):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive.")
        self.tau = float(tau)

    def forward(self, z1, z2):
        with torch.autocast(device_type=z1.device.type, enabled=False):
            u = _normalized_views(z1, z2)
            scores = (u @ u.T) / self.tau
            return _masked_cross_entropy(scores, z1.shape[0])


class AnInfoNCE(nn.Module):
    """s(u,v) = -sum_r lambda_r (u_r-v_r)^2 / (2*tau).

    ``lambda = d*softmax(raw_lambda)``, initially identity, trace d. Both
    weighted norm terms are retained. Numerical work exits autocast and uses
    at least FP32 while preserving FP64 for high precision reference tests.
    """

    def __init__(self, proj_dim=32, tau=0.07):
        super().__init__()
        if isinstance(proj_dim, bool) or not isinstance(proj_dim, int) or proj_dim < 1:
            raise ValueError("proj_dim must be a positive integer.")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive.")
        self.proj_dim = proj_dim
        self.tau = float(tau)
        self.raw_lambda = nn.Parameter(torch.zeros(proj_dim))

    def diagonal(self, dtype=None):
        raw = self.raw_lambda.to(dtype=dtype or _core_dtype(self.raw_lambda))
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Nonfinite raw_lambda.")
        lam = self.proj_dim * torch.softmax(raw, dim=0)
        if not torch.isfinite(lam).all() or torch.any(lam <= 0):
            raise FloatingPointError("AnInfoNCE weights are nonfinite or underflowed to zero.")
        return lam

    def pairwise_distances(self, u):
        """Full weighted squared distance for already normalized rows."""
        if u.ndim != 2 or u.shape[1] != self.proj_dim:
            raise ValueError("Projection dimension does not match the metric.")
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.to(_core_dtype(u, self.raw_lambda))
            lam = self.diagonal(u.dtype)
            norm2 = (u.square() * lam).sum(dim=-1)
            dist2 = norm2[:, None] + norm2[None, :] - 2 * (u * lam) @ u.T
            if not torch.isfinite(dist2).all():
                raise FloatingPointError("Nonfinite weighted distances.")
            # Bound expected accumulation error, not a change to the distance.
            tolerance = 16 * torch.finfo(u.dtype).eps * self.proj_dim
            if torch.any(dist2 < -tolerance):
                raise FloatingPointError("Weighted squared distance is significantly negative.")
            return dist2.clamp_min(0)

    def forward(self, z1, z2):
        with torch.autocast(device_type=z1.device.type, enabled=False):
            u = _normalized_views(z1, z2)
            scores = -self.pairwise_distances(u) / (2 * self.tau)
            return _masked_cross_entropy(scores, z1.shape[0])
