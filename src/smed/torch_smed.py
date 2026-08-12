"""
torch_smed.py

Core PyTorch API for balanced/unbalanced Spectral Energy Mover's Distance
(SEMD) and crossEMD.

Typical use
-----------
    loss_fn = SpectralEMDLoss(
        metric="mahalanobis",
        metric_matrix=M,
        periodic_indices=[1],
        normalize_weights=True,
        reduction="mean",
    )

    loss = loss_fn(
        points_view1, weights_view1,
        points_view2, weights_view2,
    )
    loss.backward()

Input styles
------------
1. One event:
       points1: (N1, d), weights1: (N1,)
       points2: (N2, d), weights2: (N2,)

2. Dense padded batch:
       points1: (B, N1, d), weights1: (B, N1)
       points2: (B, N2, d), weights2: (B, N2)

   Padding is represented by zero weights.

3. Variable-length batch:
       points1, weights1, points2, weights2 are lists/tuples of tensors.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from torch_smed_helpers import (
    Metric,
    balance_spectral_representations,
    build_cross_spectral_representation,
    build_double_spectral_representation,
    build_spectral_representation,
    spectral_wasserstein2_squared,
)


TensorOrSequence = Union[Tensor, Sequence[Tensor]]


def _as_event_lists(
    points: TensorOrSequence,
    weights: TensorOrSequence,
) -> tuple[list[Tensor], list[Tensor]]:
    """Convert a single event, dense batch, or list batch to event lists."""
    if isinstance(points, Tensor):
        if not isinstance(weights, Tensor):
            raise TypeError("Tensor points require tensor weights.")

        if points.ndim == 2:
            if weights.ndim != 1:
                raise ValueError(
                    "Single-event weights must have shape (N,)."
                )
            return [points], [weights]

        if points.ndim == 3:
            if weights.ndim != 2:
                raise ValueError(
                    "Batched weights must have shape (B, N)."
                )
            if points.shape[:2] != weights.shape:
                raise ValueError(
                    "For dense batches, points.shape[:2] must equal "
                    "weights.shape."
                )
            return list(points.unbind(0)), list(weights.unbind(0))

        raise ValueError(
            "Tensor points must have shape (N, d) or (B, N, d)."
        )

    if isinstance(weights, Tensor):
        raise TypeError("Sequence points require sequence weights.")

    point_list = list(points)
    weight_list = list(weights)

    if len(point_list) != len(weight_list):
        raise ValueError("Point and weight batches must have equal length.")

    return point_list, weight_list


def _reduce_losses(losses: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("`reduction` must be 'none', 'mean', or 'sum'.")


def spectral_emd(
    points1: TensorOrSequence,
    weights1: TensorOrSequence,
    points2: TensorOrSequence,
    weights2: TensorOrSequence,
    *,
    beta: float = 1.0,
    metric: Metric = "euclidean",
    metric_matrix: Optional[Tensor] = None,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Union[float, Sequence[float], Tensor] = 2.0 * torch.pi,
    normalize_weights: bool = True,
    omega_max: Optional[float] = None,
    reduction: str = "none",
    mass_rtol: float = 1e-5,
    mass_atol: float = 1e-7,
) -> Tensor:
    """
    Compute balanced or reservoir-unbalanced SEMD squared.

    Balanced mode
    -------------
    Set `omega_max=None`. The two spectral measures must have equal mass.
    Using `normalize_weights=True` guarantees unit event weight and therefore
    unit spectral mass.

    Unbalanced reservoir mode
    -------------------------
    Set `omega_max` to a finite spectral endpoint. Any missing spectral mass
    is inserted at that location before exact balanced 1D transport.

    Returns
    -------
    Tensor
        Squared SEMD values. Shape is (B,) for reduction='none', otherwise
        scalar.
    """
    p1_list, w1_list = _as_event_lists(points1, weights1)
    p2_list, w2_list = _as_event_lists(points2, weights2)

    if len(p1_list) != len(p2_list):
        raise ValueError("The two event batches must have equal batch size.")

    if omega_max is not None and omega_max < 0:
        raise ValueError("`omega_max` must be non-negative.")

    losses = []

    for p1, w1, p2, w2 in zip(p1_list, w1_list, p2_list, w2_list):
        s1 = build_spectral_representation(
            p1,
            w1,
            beta=beta,
            metric=metric,
            metric_matrix=metric_matrix,
            periodic_indices=periodic_indices,
            period=period,
            normalize=normalize_weights,
        )
        s2 = build_spectral_representation(
            p2,
            w2,
            beta=beta,
            metric=metric,
            metric_matrix=metric_matrix,
            periodic_indices=periodic_indices,
            period=period,
            normalize=normalize_weights,
        )

        if omega_max is not None:
            s1, s2 = balance_spectral_representations(
                s1,
                s2,
                omega_max,
                atol=mass_atol,
            )

        loss = spectral_wasserstein2_squared(
            s1,
            s2,
            check_mass=True,
            rtol=mass_rtol,
            atol=mass_atol,
        )
        losses.append(loss)

    stacked = torch.stack(losses)
    return _reduce_losses(stacked, reduction)


def cross_emd(
    points1: TensorOrSequence,
    weights1: TensorOrSequence,
    points2: TensorOrSequence,
    weights2: TensorOrSequence,
    *,
    beta: float = 1.0,
    metric: str = "euclidean",
    normalize_weights: bool = True,
    reduction: str = "none",
) -> Tensor:
    """
    Compute SPECTER-style crossEMD squared.

    It compares:
        s_within = s(event1) + s(event2)
    against:
        s_cross  = spectrum of every event1-event2 pair.

    This is generally sensitive to relative alignment and should not be
    confused with ordinary independently rotation-invariant SEMD.
    """
    p1_list, w1_list = _as_event_lists(points1, weights1)
    p2_list, w2_list = _as_event_lists(points2, weights2)

    if len(p1_list) != len(p2_list):
        raise ValueError("The two event batches must have equal batch size.")

    losses = []

    for p1, w1, p2, w2 in zip(p1_list, w1_list, p2_list, w2_list):
        within = build_double_spectral_representation(
            p1,
            w1,
            p2,
            w2,
            beta=beta,
            metric=metric,
            normalize=normalize_weights,
        )
        cross = build_cross_spectral_representation(
            p1,
            w1,
            p2,
            w2,
            beta=beta,
            metric=metric,
            normalize=normalize_weights,
        )
        losses.append(
            spectral_wasserstein2_squared(within, cross, check_mass=True)
        )

    stacked = torch.stack(losses)
    return _reduce_losses(stacked, reduction)


class SpectralEMDLoss(nn.Module):
    """
    nn.Module wrapper around `spectral_emd`.

    The returned quantity is squared SEMD. This is normally preferable as a
    training loss because taking a square root creates a singular derivative
    at zero.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        metric: Metric = "euclidean",
        metric_matrix: Optional[Tensor] = None,
        periodic_indices: Optional[Sequence[int]] = None,
        period: Union[float, Sequence[float], Tensor] = 2.0 * torch.pi,
        normalize_weights: bool = True,
        omega_max: Optional[float] = None,
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
        points1: TensorOrSequence,
        weights1: TensorOrSequence,
        points2: TensorOrSequence,
        weights2: TensorOrSequence,
    ) -> Tensor:
        return spectral_emd(
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


class CrossEMDLoss(nn.Module):
    """nn.Module wrapper around SPECTER-style `cross_emd`."""

    def __init__(
        self,
        *,
        beta: float = 1.0,
        metric: str = "euclidean",
        normalize_weights: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.metric = metric
        self.normalize_weights = normalize_weights
        self.reduction = reduction

    def forward(
        self,
        points1: TensorOrSequence,
        weights1: TensorOrSequence,
        points2: TensorOrSequence,
        weights2: TensorOrSequence,
    ) -> Tensor:
        return cross_emd(
            points1,
            weights1,
            points2,
            weights2,
            beta=self.beta,
            metric=self.metric,
            normalize_weights=self.normalize_weights,
            reduction=self.reduction,
        )
