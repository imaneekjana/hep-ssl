"""The shared objective for single-space and five-space ablations."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .contrastive import AnInfoNCE, CosineInfoNCE, _core_dtype


SPACE_ORDER = ("general", "energy", "eta", "phi", "local")
TASKS = ("energy", "eta", "phi", "local")
MODES = (
    "single_cosine", "single_anisotropic", "five_anisotropic_no_aux",
    "five_anisotropic_physics",
)


class MultiTaskObjective(nn.Module):
    def __init__(self, mode="five_anisotropic_physics", tau=0.07, gamma=1.0, proj_dim=32):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown objective mode {mode!r}.")
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative.")
        self.mode = mode
        self.tau = float(tau)
        self.gamma = float(gamma) if mode == "five_anisotropic_physics" else 0.0
        self.space_order = ("general",) if mode.startswith("single_") else SPACE_ORDER
        self.cl_losses = nn.ModuleDict({
            name: (CosineInfoNCE(tau) if mode == "single_cosine" else AnInfoNCE(proj_dim, tau))
            for name in self.space_order
        })

    def metric_diagonals(self):
        return {name: loss.diagonal() for name, loss in self.cl_losses.items()
                if isinstance(loss, AnInfoNCE)}

    @staticmethod
    def _task_loss(name, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shapes differ for {name}.")
        # Labels are fixed data; objectives must not create a target-side graph.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            dtype = _core_dtype(prediction, target)
            prediction = prediction.to(dtype)
            target = target.detach().to(device=prediction.device, dtype=dtype)
            if not torch.isfinite(target).all():
                raise FloatingPointError(f"Nonfinite {name} target.")
            if name == "eta":
                if torch.any(target < 0) or not torch.allclose(
                    target.sum(dim=-1), torch.ones_like(target[:, 0]), atol=1e-5, rtol=1e-5
                ):
                    raise ValueError("eta targets must be probability distributions.")
                return F.kl_div(F.log_softmax(prediction, dim=-1), target, reduction="batchmean")
            return F.mse_loss(prediction, target, reduction="mean")

    def forward(self, out1, out2, targets1=None, targets2=None):
        parts = {f"cl/{name}": loss(out1["z"][name], out2["z"][name])
                 for name, loss in self.cl_losses.items()}
        contrastive = torch.stack(list(parts.values())).mean()
        physics = contrastive.new_zeros(())
        if self.gamma > 0:
            if targets1 is None or targets2 is None:
                raise ValueError("Physics mode requires each view's own reference targets.")
            for name in TASKS:
                one = self._task_loss(name, out1["pred"][name], targets1[name])
                two = self._task_loss(name, out2["pred"][name], targets2[name])
                parts[f"physics/{name}"] = 0.5 * (one + two)
            physics = torch.stack([parts[f"physics/{name}"] for name in TASKS]).mean()
        return {"loss": contrastive + self.gamma * physics,
                "contrastive": contrastive, "physics": physics, **parts}
