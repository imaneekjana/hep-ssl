"""
vectorized_smed.py

Fully vectorized PyTorch SEMD core functions and nn.Module wrappers.

This implementation accepts padded dense batches only:

    points1:  (B, N1, d)
    weights1: (B, N1)
    points2:  (B, N2, d)
    weights2: (B, N2)

Set padded weights exactly to zero. N1 and N2 may differ.

The returned loss is squared SEMD, which is normally preferable for training.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn

from vectorized_smed_helpers import (
    Metric,
    Period,
    append_reservoir_atoms,
    batched_wasserstein2_squared,
    build_batched_spectral_representation,
)


def _reduce(losses: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("`reduction` must be 'none', 'mean', or 'sum'.")


def vectorized_spectral_emd(
    points1: Tensor,
    weights1: Tensor,
    points2: Tensor,
    weights2: Tensor,
    *,
    beta: float = 1.0,
    metric: Metric = "euclidean",
    metric_matrix: Optional[Tensor] = None,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Period = 2.0 * torch.pi,
    normalize_weights: bool = True,
    omega_max: Optional[Union[float, Tensor]] = None,
    reduction: str = "none",
    mass_rtol: float = 1e-5,
    mass_atol: float = 1e-7,
) -> Tensor:
    """
    Compute balanced or reservoir-unbalanced squared SEMD for a padded batch.

    Balanced mode
    -------------
    Use `omega_max=None`. Equal total event weights are required unless
    `normalize_weights=True`, which normalizes every event to total weight one.

    Unbalanced reservoir mode
    -------------------------
    Supply `omega_max`. The smaller spectral measure receives its missing mass
    at that spectral location before exact 1D transport.

    Returns
    -------
    losses : Tensor
        Shape (B,) when reduction='none', otherwise a scalar.
    """
    spectral1 = build_batched_spectral_representation(
        points1,
        weights1,
        beta=beta,
        metric=metric,
        metric_matrix=metric_matrix,
        periodic_indices=periodic_indices,
        period=period,
        normalize_weights=normalize_weights,
    )
    spectral2 = build_batched_spectral_representation(
        points2,
        weights2,
        beta=beta,
        metric=metric,
        metric_matrix=metric_matrix,
        periodic_indices=periodic_indices,
        period=period,
        normalize_weights=normalize_weights,
    )

    if omega_max is not None:
        spectral1, spectral2 = append_reservoir_atoms(
            spectral1,
            spectral2,
            omega_max,
        )

    losses = batched_wasserstein2_squared(
        spectral1,
        spectral2,
        check_mass=True,
        rtol=mass_rtol,
        atol=mass_atol,
    )
    return _reduce(losses, reduction)


class VectorizedSpectralEMDLoss(nn.Module):
    """PyTorch loss module for fully vectorized padded-batch SEMD."""

    def __init__(
        self,
        *,
        beta: float = 1.0,
        metric: Metric = "euclidean",
        metric_matrix: Optional[Tensor] = None,
        periodic_indices: Optional[Sequence[int]] = None,
        period: Period = 2.0 * torch.pi,
        normalize_weights: bool = True,
        omega_max: Optional[Union[float, Tensor]] = None,
        reduction: str = "mean",
        mass_rtol: float = 1e-5,
        mass_atol: float = 1e-7,
    ) -> None:
        super().__init__()

        if beta <= 0:
            raise ValueError("`beta` must be positive.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("Invalid reduction.")

        self.beta = beta
        self.metric = metric
        self.periodic_indices = periodic_indices
        self.period = period
        self.normalize_weights = normalize_weights
        self.omega_max = omega_max
        self.reduction = reduction
        self.mass_rtol = mass_rtol
        self.mass_atol = mass_atol

        if metric_matrix is None:
            self.register_buffer("metric_matrix", None)
        else:
            self.register_buffer(
                "metric_matrix",
                torch.as_tensor(metric_matrix),
            )

    def forward(
        self,
        points1: Tensor,
        weights1: Tensor,
        points2: Tensor,
        weights2: Tensor,
    ) -> Tensor:
        return vectorized_spectral_emd(
            points1,
            weights1,
            points2,
            weights2,
            beta=self.beta,
            metric=self.metric,
            metric_matrix=self.metric_matrix,
            periodic_indices=self.periodic_indices,
            period=self.period,
            normalize_weights=self.normalize_weights,
            omega_max=self.omega_max,
            reduction=self.reduction,
            mass_rtol=self.mass_rtol,
            mass_atol=self.mass_atol,
        )
