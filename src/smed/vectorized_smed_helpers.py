"""
vectorized_smed_helpers.py

Fully vectorized PyTorch helpers for Spectral Energy Mover's Distance (SEMD).

Required padded-batch format
----------------------------
points:  (B, N, d)
weights: (B, N)

Use exactly zero weight for padded points. Different maximum point counts are
allowed for the two views, e.g. N1 != N2.

The spectral representation for each event contains:
  * one atom at omega=0 with mass sum_i w_i^2;
  * one atom for each unordered pair i<j with mass 2 w_i w_j.

For a squared base metric D_ij, the spectral coordinate is

    omega_ij = D_ij ** (beta / 2) / beta.

At beta=1:
  * squared Euclidean input produces Euclidean spectral distances;
  * Delta-R squared produces Delta-R;
  * (1-cos(theta))^2 produces 1-cos(theta).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

Metric = Union[str, Callable[[Tensor], Tensor]]
Period = Union[float, Sequence[float], Tensor]


def validate_batched_event(points: Tensor, weights: Tensor) -> None:
    """Validate one padded batch of weighted point clouds."""
    if points.ndim != 3:
        raise ValueError(
            f"`points` must have shape (B, N, d); got {tuple(points.shape)}."
        )
    if weights.ndim != 2:
        raise ValueError(
            f"`weights` must have shape (B, N); got {tuple(weights.shape)}."
        )
    if points.shape[:2] != weights.shape:
        raise ValueError(
            "`weights.shape` must equal `points.shape[:2]`; got "
            f"{tuple(weights.shape)} and {tuple(points.shape[:2])}."
        )
    if not points.is_floating_point() or not weights.is_floating_point():
        raise TypeError("`points` and `weights` must be floating-point tensors.")
    if points.device != weights.device:
        raise ValueError("`points` and `weights` must be on the same device.")
    if torch.any(weights < 0):
        raise ValueError("SEMD weights must be non-negative.")


def normalize_batched_weights(weights: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize each event's weights to unit sum."""
    totals = weights.sum(dim=1, keepdim=True)
    if torch.any(totals <= eps):
        raise ValueError("Every event must have positive total weight.")
    return weights / totals.clamp_min(eps)


def wrap_periodic_differences(
    differences: Tensor,
    periodic_indices: Optional[Sequence[int]],
    period: Period = 2.0 * torch.pi,
) -> Tensor:
    """
    Wrap selected final-axis coordinates to their shortest signed displacement.

    `differences` may have any leading dimensions and final dimension d.
    """
    if not periodic_indices:
        return differences

    d = differences.shape[-1]
    indices = torch.as_tensor(
        list(periodic_indices),
        device=differences.device,
        dtype=torch.long,
    )
    if torch.any(indices < 0) or torch.any(indices >= d):
        raise IndexError("Every periodic index must satisfy 0 <= index < d.")

    periods = torch.as_tensor(
        period,
        device=differences.device,
        dtype=differences.dtype,
    )
    if periods.ndim == 0:
        periods = periods.expand(indices.numel())
    if periods.shape != indices.shape:
        raise ValueError(
            "`period` must be scalar or contain one value per periodic index."
        )
    if torch.any(periods <= 0):
        raise ValueError("All periods must be positive.")

    selected = differences.index_select(-1, indices)
    angles = (2.0 * torch.pi / periods) * selected
    wrapped = (
        periods
        / (2.0 * torch.pi)
        * torch.atan2(torch.sin(angles), torch.cos(angles))
    )

    result = differences.clone()
    result[..., indices] = wrapped
    return result


def batched_euclidean_metric(points: Tensor) -> Tensor:
    """Pairwise squared Euclidean distances, shape (B, N, N)."""
    diff = points[:, :, None, :] - points[:, None, :, :]
    return diff.square().sum(dim=-1)


def batched_cylindrical_metric(
    points: Tensor,
    eta_index: int = 0,
    phi_index: int = 1,
) -> Tensor:
    """Pairwise Delta-R squared, shape (B, N, N)."""
    d = points.shape[-1]
    if not (0 <= eta_index < d and 0 <= phi_index < d):
        raise IndexError("`eta_index` and `phi_index` must lie in [0, d).")

    delta_eta = points[:, :, None, eta_index] - points[:, None, :, eta_index]
    raw_phi = points[:, :, None, phi_index] - points[:, None, :, phi_index]
    delta_phi = torch.atan2(torch.sin(raw_phi), torch.cos(raw_phi))
    return delta_eta.square() + delta_phi.square()


def batched_spherical_metric(points: Tensor, eps: float = 1e-12) -> Tensor:
    """
    SPECTER-style squared spherical quantity:

        D_ij = (1 - cos(theta_ij))^2.
    """
    norms = torch.linalg.vector_norm(points, dim=-1, keepdim=True).clamp_min(eps)
    unit = points / norms
    cosine = torch.einsum("bid,bjd->bij", unit, unit).clamp(-1.0, 1.0)
    return (1.0 - cosine).square()


def batched_mahalanobis_metric(
    points: Tensor,
    matrix: Tensor,
    *,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Period = 2.0 * torch.pi,
    symmetrize_matrix: bool = True,
) -> Tensor:
    """
    Pairwise squared Mahalanobis-style distances.

    `matrix` may have shape:
      * (d, d): one metric shared across the batch;
      * (B, d, d): one metric per event.

    For a genuine distance, each matrix should be symmetric positive
    semidefinite.
    """
    batch_size, _, d = points.shape
    matrix = torch.as_tensor(matrix, device=points.device, dtype=points.dtype)

    if matrix.ndim == 2:
        if matrix.shape != (d, d):
            raise ValueError(f"Shared matrix must have shape ({d}, {d}).")
        if symmetrize_matrix:
            matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    elif matrix.ndim == 3:
        if matrix.shape != (batch_size, d, d):
            raise ValueError(
                f"Batched matrix must have shape ({batch_size}, {d}, {d})."
            )
        if symmetrize_matrix:
            matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    else:
        raise ValueError("`matrix` must have shape (d,d) or (B,d,d).")

    diff = points[:, :, None, :] - points[:, None, :, :]
    diff = wrap_periodic_differences(diff, periodic_indices, period)

    if matrix.ndim == 2:
        distance2 = torch.einsum("bnmi,ij,bnmj->bnm", diff, matrix, diff)
    else:
        distance2 = torch.einsum("bnmi,bij,bnmj->bnm", diff, matrix, diff)

    return distance2.clamp_min(0.0)


def batched_metric_matrix(
    points: Tensor,
    metric: Metric = "euclidean",
    *,
    metric_matrix: Optional[Tensor] = None,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Period = 2.0 * torch.pi,
) -> Tensor:
    """Dispatch a named metric or a custom batched metric callable."""
    if callable(metric):
        result = metric(points)
    elif metric == "euclidean":
        result = batched_euclidean_metric(points)
    elif metric == "cylindrical":
        result = batched_cylindrical_metric(points)
    elif metric == "spherical":
        result = batched_spherical_metric(points)
    elif metric == "mahalanobis":
        if metric_matrix is None:
            raise ValueError(
                "`metric_matrix` is required when metric='mahalanobis'."
            )
        result = batched_mahalanobis_metric(
            points,
            metric_matrix,
            periodic_indices=periodic_indices,
            period=period,
        )
    else:
        raise ValueError(
            "Unknown metric. Use 'euclidean', 'cylindrical', 'spherical', "
            "'mahalanobis', or pass a callable."
        )

    expected = (points.shape[0], points.shape[1], points.shape[1])
    if result.shape != expected:
        raise ValueError(
            f"Metric callable must return shape {expected}; got {result.shape}."
        )
    return result


def build_batched_spectral_representation(
    points: Tensor,
    weights: Tensor,
    *,
    beta: float = 1.0,
    metric: Metric = "euclidean",
    metric_matrix: Optional[Tensor] = None,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Period = 2.0 * torch.pi,
    normalize_weights: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """
    Build sorted spectral measures for an entire padded batch.

    Returns
    -------
    spectral : Tensor, shape (B, M, 2)
        M = 1 + N(N-1)/2.
        spectral[..., 0] contains omega locations.
        spectral[..., 1] contains masses.
    """
    validate_batched_event(points, weights)
    if beta <= 0:
        raise ValueError("`beta` must be positive.")

    if normalize_weights:
        weights = normalize_batched_weights(weights, eps=eps)

    batch_size, n_points, _ = points.shape
    distance2 = batched_metric_matrix(
        points,
        metric,
        metric_matrix=metric_matrix,
        periodic_indices=periodic_indices,
        period=period,
    )

    if n_points < 2:
        pair_omega = points.new_empty((batch_size, 0))
        pair_mass = weights.new_empty((batch_size, 0))
    else:
        row, col = torch.triu_indices(
            n_points,
            n_points,
            offset=1,
            device=points.device,
        )
        pair_distance2 = distance2[:, row, col].clamp_min(0.0)
        pair_mass = 2.0 * weights[:, row] * weights[:, col]
        pair_omega = pair_distance2.pow(beta / 2.0) / beta

        # Padding atoms remain inert and are placed at omega=0.
        pair_omega = torch.where(
            pair_mass > 0,
            pair_omega,
            torch.zeros_like(pair_omega),
        )

        pair_omega, order = torch.sort(pair_omega, dim=1)
        pair_mass = torch.gather(pair_mass, 1, order)

    self_mass = weights.square().sum(dim=1, keepdim=True)
    zero = points.new_zeros((batch_size, 1))

    omega = torch.cat((zero, pair_omega), dim=1)
    mass = torch.cat((self_mass, pair_mass), dim=1)
    return torch.stack((omega, mass), dim=-1)


def append_reservoir_atoms(
    spectral1: Tensor,
    spectral2: Tensor,
    omega_max: Union[float, Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Vectorized reservoir balancing for unequal spectral masses.

    One atom is appended to each spectrum:
      * the lower-mass spectrum receives its deficit at omega_max;
      * the higher-mass spectrum receives a zero-mass atom at omega=0.

    The result is then independently sorted per batch item.
    """
    if spectral1.ndim != 3 or spectral1.shape[-1] != 2:
        raise ValueError("`spectral1` must have shape (B, M1, 2).")
    if spectral2.ndim != 3 or spectral2.shape[-1] != 2:
        raise ValueError("`spectral2` must have shape (B, M2, 2).")
    if spectral1.shape[0] != spectral2.shape[0]:
        raise ValueError("The spectra must have the same batch size.")

    total1 = spectral1[..., 1].sum(dim=1)
    total2 = spectral2[..., 1].sum(dim=1)
    deficit1 = (total2 - total1).clamp_min(0.0)
    deficit2 = (total1 - total2).clamp_min(0.0)

    omega_max_tensor = torch.as_tensor(
        omega_max,
        device=spectral1.device,
        dtype=spectral1.dtype,
    )
    if torch.any(omega_max_tensor < 0):
        raise ValueError("`omega_max` must be non-negative.")

    omega_max_tensor = omega_max_tensor.expand_as(deficit1)
    zero1 = torch.zeros_like(deficit1)
    zero2 = torch.zeros_like(deficit2)

    atom1_omega = torch.where(deficit1 > 0, omega_max_tensor, zero1)
    atom2_omega = torch.where(deficit2 > 0, omega_max_tensor, zero2)

    atom1 = torch.stack((atom1_omega, deficit1), dim=-1).unsqueeze(1)
    atom2 = torch.stack((atom2_omega, deficit2), dim=-1).unsqueeze(1)

    out1 = torch.cat((spectral1, atom1), dim=1)
    out2 = torch.cat((spectral2, atom2), dim=1)

    order1 = torch.argsort(out1[..., 0], dim=1)
    order2 = torch.argsort(out2[..., 0], dim=1)
    out1 = torch.gather(out1, 1, order1[..., None].expand(-1, -1, 2))
    out2 = torch.gather(out2, 1, order2[..., None].expand(-1, -1, 2))
    return out1, out2


def batched_wasserstein2_squared(
    spectral1: Tensor,
    spectral2: Tensor,
    *,
    check_mass: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tensor:
    """
    Exact vectorized 1D W2^2 between discrete batched measures.

    The method forms the union of cumulative-mass endpoints. Between adjacent
    endpoints, each inverse CDF is constant, so the integral of their squared
    difference is exact.

    Returns shape (B,).
    """
    if spectral1.ndim != 3 or spectral1.shape[-1] != 2:
        raise ValueError("`spectral1` must have shape (B, M1, 2).")
    if spectral2.ndim != 3 or spectral2.shape[-1] != 2:
        raise ValueError("`spectral2` must have shape (B, M2, 2).")
    if spectral1.shape[0] != spectral2.shape[0]:
        raise ValueError("The spectra must have the same batch size.")

    omega1, mass1 = spectral1[..., 0], spectral1[..., 1]
    omega2, mass2 = spectral2[..., 0], spectral2[..., 1]

    if torch.any(mass1 < 0) or torch.any(mass2 < 0):
        raise ValueError("Spectral masses must be non-negative.")

    total1 = mass1.sum(dim=1)
    total2 = mass2.sum(dim=1)
    if check_mass and not torch.allclose(total1, total2, rtol=rtol, atol=atol):
        max_error = (total1 - total2).abs().max().detach().cpu().item()
        raise ValueError(
            "Balanced transport requires equal total spectral mass. "
            f"Maximum batch mismatch: {max_error:.6g}."
        )

    cumulative1 = torch.cumsum(mass1, dim=1)
    cumulative2 = torch.cumsum(mass2, dim=1)

    zero = torch.zeros(
        (spectral1.shape[0], 1),
        device=spectral1.device,
        dtype=spectral1.dtype,
    )
    boundaries = torch.cat((zero, cumulative1, cumulative2), dim=1)
    boundaries, _ = torch.sort(boundaries, dim=1)

    left = boundaries[:, :-1]
    right = boundaries[:, 1:]
    interval_mass = (right - left).clamp_min(0.0)
    midpoint = 0.5 * (left + right)

    # Batched searchsorted determines which spectral atom owns each interval.
    index1 = torch.searchsorted(cumulative1.contiguous(), midpoint.contiguous())
    index2 = torch.searchsorted(cumulative2.contiguous(), midpoint.contiguous())
    index1 = index1.clamp_max(omega1.shape[1] - 1)
    index2 = index2.clamp_max(omega2.shape[1] - 1)

    quantile1 = torch.gather(omega1, 1, index1)
    quantile2 = torch.gather(omega2, 1, index2)

    return (interval_mass * (quantile1 - quantile2).square()).sum(dim=1)
