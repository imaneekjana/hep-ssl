"""
torch_smed_helpers.py

Differentiable PyTorch helper functions for Spectral Energy Mover's Distance
(SEMD), adapted from the mathematical structure used by Rikab Gambhir's
JAX-based SPECTER package.

Conventions
-----------
An event is represented by:
    points:  Tensor[N, d]
    weights: Tensor[N]

The spectral representation is a Tensor[M, 2]:
    spectral[:, 0] = omega locations
    spectral[:, 1] = non-negative spectral masses

For N points, M = 1 + N(N-1)/2:
    - one atom at omega = 0 with mass sum_i w_i^2
    - one atom per unordered pair i < j with mass 2 w_i w_j

The spectral location for a pair is
    omega_ij = metric_squared(x_i, x_j) ** (beta / 2) / beta

Thus, when beta=1:
    - squared Euclidean input gives ordinary Euclidean distance;
    - squared cylindrical input gives Delta-R;
    - the SPECTER-style spherical metric gives 1 - cos(theta).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union

import torch
from torch import Tensor

Metric = Union[str, Callable[[Tensor], Tensor]]


def _validate_points(points: Tensor) -> None:
    if points.ndim != 2:
        raise ValueError(
            f"`points` must have shape (N, d); received {tuple(points.shape)}."
        )
    if not points.is_floating_point():
        raise TypeError("`points` must be a floating-point tensor.")


def _validate_points_and_weights(points: Tensor, weights: Tensor) -> None:
    _validate_points(points)

    if weights.ndim != 1:
        raise ValueError(
            f"`weights` must have shape (N,); received {tuple(weights.shape)}."
        )
    if points.shape[0] != weights.shape[0]:
        raise ValueError(
            "The number of points and weights must agree: "
            f"{points.shape[0]} != {weights.shape[0]}."
        )
    if not weights.is_floating_point():
        raise TypeError("`weights` must be a floating-point tensor.")
    if points.device != weights.device:
        raise ValueError("`points` and `weights` must be on the same device.")
    if torch.any(weights < 0):
        raise ValueError("SEMD requires non-negative weights.")


def normalize_weights(weights: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize non-negative event weights to sum to one."""
    if weights.ndim != 1:
        raise ValueError("`weights` must have shape (N,).")
    total = weights.sum()
    if bool((total <= eps).detach().cpu()):
        raise ValueError("Cannot normalize an event with zero total weight.")
    return weights / total.clamp_min(eps)


def euclidean_metric(points: Tensor) -> Tensor:
    """Return pairwise squared Euclidean distances, shape (N, N)."""
    _validate_points(points)
    diff = points[:, None, :] - points[None, :, :]
    return (diff * diff).sum(dim=-1)


def cylindrical_metric(
    points: Tensor,
    eta_index: int = 0,
    phi_index: int = 1,
) -> Tensor:
    """
    Return pairwise Delta-R squared for coordinates containing (eta, phi).

    D_ij = (eta_i - eta_j)^2 + wrapped(phi_i - phi_j)^2.
    """
    _validate_points(points)
    d = points.shape[1]

    if not (0 <= eta_index < d and 0 <= phi_index < d):
        raise IndexError("`eta_index` and `phi_index` must index point features.")

    delta_eta = points[:, None, eta_index] - points[None, :, eta_index]
    raw_delta_phi = points[:, None, phi_index] - points[None, :, phi_index]

    # Signed shortest displacement on the circle, in [-pi, pi].
    delta_phi = torch.atan2(torch.sin(raw_delta_phi), torch.cos(raw_delta_phi))

    return delta_eta.square() + delta_phi.square()


def spherical_metric(points: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Return the squared SPECTER-style spherical quantity

        D_ij = (1 - cos(theta_ij))^2.

    The spectral builder later raises D_ij to beta/2. Therefore beta=1
    gives omega_ij = 1 - cos(theta_ij).
    """
    _validate_points(points)

    norms = torch.linalg.vector_norm(points, dim=-1).clamp_min(eps)
    unit = points / norms[:, None]
    cosine = unit @ unit.transpose(0, 1)
    cosine = cosine.clamp(-1.0, 1.0)

    return (1.0 - cosine).square()


def mahalanobis_metric(
    points: Tensor,
    matrix: Tensor,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Union[float, Sequence[float], Tensor] = 2.0 * torch.pi,
    symmetrize_matrix: bool = True,
) -> Tensor:
    """
    Return pairwise squared Mahalanobis-style distances

        D_ij = delta_ij^T M delta_ij,

    optionally wrapping selected coordinate differences periodically.

    Notes
    -----
    For this to define a non-negative squared distance, M should be symmetric
    positive semidefinite. If `symmetrize_matrix=True`, M is replaced by
    (M + M.T)/2, but positive semidefiniteness is still the caller's
    responsibility.
    """
    _validate_points(points)

    d = points.shape[1]
    matrix = torch.as_tensor(
        matrix,
        device=points.device,
        dtype=points.dtype,
    )

    if matrix.shape != (d, d):
        raise ValueError(
            f"`matrix` must have shape ({d}, {d}); got {tuple(matrix.shape)}."
        )

    if symmetrize_matrix:
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))

    diff = points[:, None, :] - points[None, :, :]

    if periodic_indices:
        indices = torch.as_tensor(
            list(periodic_indices),
            device=points.device,
            dtype=torch.long,
        )

        if torch.any(indices < 0) or torch.any(indices >= d):
            raise IndexError("Every periodic index must lie in [0, d).")

        periods = torch.as_tensor(
            period,
            device=points.device,
            dtype=points.dtype,
        )
        if periods.ndim == 0:
            periods = periods.repeat(indices.numel())
        if periods.shape != indices.shape:
            raise ValueError(
                "`period` must be scalar or have one value per periodic index."
            )
        if torch.any(periods <= 0):
            raise ValueError("All periods must be positive.")

        raw = diff.index_select(dim=-1, index=indices)
        angles = (2.0 * torch.pi / periods) * raw
        wrapped = (
            periods
            / (2.0 * torch.pi)
            * torch.atan2(torch.sin(angles), torch.cos(angles))
        )

        # Avoid an in-place write on a tensor needed by autograd.
        wrapped_diff = diff.clone()
        wrapped_diff[..., indices] = wrapped
        diff = wrapped_diff

    distances_squared = torch.einsum(
        "...i,ij,...j->...",
        diff,
        matrix,
        diff,
    )

    # Clamp only guards against small negative roundoff when M is PSD.
    return distances_squared.clamp_min(0.0)


def pairwise_metric_matrix(
    points: Tensor,
    metric: Metric = "euclidean",
    *,
    metric_matrix: Optional[Tensor] = None,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Union[float, Sequence[float], Tensor] = 2.0 * torch.pi,
) -> Tensor:
    """Dispatch a named metric or call a custom metric function."""
    if callable(metric):
        result = metric(points)
    elif metric == "euclidean":
        result = euclidean_metric(points)
    elif metric == "cylindrical":
        result = cylindrical_metric(points)
    elif metric == "spherical":
        result = spherical_metric(points)
    elif metric == "mahalanobis":
        if metric_matrix is None:
            raise ValueError(
                "`metric_matrix` is required when metric='mahalanobis'."
            )
        result = mahalanobis_metric(
            points,
            metric_matrix,
            periodic_indices=periodic_indices,
            period=period,
        )
    else:
        raise ValueError(
            "Unknown metric. Choose 'euclidean', 'cylindrical', "
            "'spherical', 'mahalanobis', or pass a callable."
        )

    expected = (points.shape[0], points.shape[0])
    if result.shape != expected:
        raise ValueError(
            f"Metric must return shape {expected}; got {tuple(result.shape)}."
        )
    return result


def cross_metric_matrix(
    points1: Tensor,
    points2: Tensor,
    metric: str = "euclidean",
    *,
    eps: float = 1e-12,
) -> Tensor:
    """
    Return cross-event squared metric values, shape (N1, N2).

    Currently supports Euclidean and SPECTER-style spherical metrics.
    """
    _validate_points(points1)
    _validate_points(points2)

    if points1.shape[1] != points2.shape[1]:
        raise ValueError("Both point sets must have the same feature dimension.")
    if points1.device != points2.device:
        raise ValueError("Both point sets must be on the same device.")

    if metric == "euclidean":
        diff = points1[:, None, :] - points2[None, :, :]
        return (diff * diff).sum(dim=-1)

    if metric == "spherical":
        norm1 = torch.linalg.vector_norm(points1, dim=-1).clamp_min(eps)
        norm2 = torch.linalg.vector_norm(points2, dim=-1).clamp_min(eps)
        unit1 = points1 / norm1[:, None]
        unit2 = points2 / norm2[:, None]
        cosine = (unit1[:, None, :] * unit2[None, :, :]).sum(dim=-1)
        cosine = cosine.clamp(-1.0, 1.0)
        return (1.0 - cosine).square()

    raise ValueError("Cross metrics currently support 'euclidean' or 'spherical'.")


def build_spectral_representation(
    points: Tensor,
    weights: Tensor,
    *,
    beta: float = 1.0,
    metric: Metric = "euclidean",
    metric_matrix: Optional[Tensor] = None,
    periodic_indices: Optional[Sequence[int]] = None,
    period: Union[float, Sequence[float], Tensor] = 2.0 * torch.pi,
    normalize: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """
    Build one sorted spectral representation, shape (1 + N(N-1)/2, 2).

    Padding points are supported by assigning them exactly zero weight.
    """
    _validate_points_and_weights(points, weights)

    if beta <= 0:
        raise ValueError("`beta` must be positive.")

    if normalize:
        weights = normalize_weights(weights, eps=eps)

    n = points.shape[0]

    distance_squared = pairwise_metric_matrix(
        points,
        metric,
        metric_matrix=metric_matrix,
        periodic_indices=periodic_indices,
        period=period,
    )

    if n < 2:
        pair_omega = points.new_empty((0,))
        pair_mass = weights.new_empty((0,))
    else:
        i, j = torch.triu_indices(n, n, offset=1, device=points.device)
        pair_distance_squared = distance_squared[i, j].clamp_min(0.0)

        pair_omega = pair_distance_squared.pow(beta / 2.0) / beta
        pair_mass = 2.0 * weights[i] * weights[j]

        # Zero-weight padding should produce an inert atom at omega=0.
        pair_omega = torch.where(
            pair_mass > 0,
            pair_omega,
            torch.zeros_like(pair_omega),
        )

        order = torch.argsort(pair_omega)
        pair_omega = pair_omega[order]
        pair_mass = pair_mass[order]

    self_mass = weights.square().sum().reshape(1)
    zero = points.new_zeros((1,))

    omega = torch.cat((zero, pair_omega), dim=0)
    mass = torch.cat((self_mass, pair_mass), dim=0)

    return torch.stack((omega, mass), dim=-1)


def build_double_spectral_representation(
    points1: Tensor,
    weights1: Tensor,
    points2: Tensor,
    weights2: Tensor,
    **kwargs,
) -> Tensor:
    """Concatenate and sort the two ordinary spectral representations."""
    s1 = build_spectral_representation(points1, weights1, **kwargs)
    s2 = build_spectral_representation(points2, weights2, **kwargs)
    spectral = torch.cat((s1, s2), dim=0)
    return spectral[torch.argsort(spectral[:, 0])]


def build_cross_spectral_representation(
    points1: Tensor,
    weights1: Tensor,
    points2: Tensor,
    weights2: Tensor,
    *,
    beta: float = 1.0,
    metric: str = "euclidean",
    normalize: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """
    Build the cross spectral representation with atoms

        omega_ij = d(x_i, y_j)^beta / beta
        mass_ij  = 2 w_i v_j.
    """
    _validate_points_and_weights(points1, weights1)
    _validate_points_and_weights(points2, weights2)

    if beta <= 0:
        raise ValueError("`beta` must be positive.")

    if normalize:
        weights1 = normalize_weights(weights1, eps=eps)
        weights2 = normalize_weights(weights2, eps=eps)

    distance_squared = cross_metric_matrix(points1, points2, metric=metric)
    omega = distance_squared.clamp_min(0.0).pow(beta / 2.0) / beta
    mass = 2.0 * weights1[:, None] * weights2[None, :]

    omega = torch.where(mass > 0, omega, torch.zeros_like(omega))
    omega = omega.reshape(-1)
    mass = mass.reshape(-1)

    order = torch.argsort(omega)
    return torch.stack((omega[order], mass[order]), dim=-1)


def augment_spectral_representation(
    spectral: Tensor,
    omega: Union[float, Tensor],
    added_mass: Tensor,
) -> Tensor:
    """Insert one spectral atom and return the result sorted by omega."""
    if spectral.ndim != 2 or spectral.shape[1] != 2:
        raise ValueError("`spectral` must have shape (M, 2).")

    omega_tensor = torch.as_tensor(
        omega,
        device=spectral.device,
        dtype=spectral.dtype,
    ).reshape(1)
    mass_tensor = torch.as_tensor(
        added_mass,
        device=spectral.device,
        dtype=spectral.dtype,
    ).reshape(1)

    atom = torch.stack((omega_tensor, mass_tensor), dim=-1)
    augmented = torch.cat((spectral, atom), dim=0)
    return augmented[torch.argsort(augmented[:, 0])]


def balance_spectral_representations(
    spectral1: Tensor,
    spectral2: Tensor,
    omega_max: Union[float, Tensor],
    *,
    atol: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """
    Balance unequal spectral masses by adding the deficit at omega_max.

    A zero-mass atom is added to the other spectrum so both outputs also have
    the same number of atoms, matching SPECTER's balancing convention.
    """
    total1 = spectral1[:, 1].sum()
    total2 = spectral2[:, 1].sum()
    difference = total1 - total2

    # This branch is non-smooth only where total1 == total2, as expected for
    # the unbalanced reservoir construction.
    difference_value = float(difference.detach().cpu())

    if abs(difference_value) <= atol:
        return spectral1, spectral2

    zero_mass = difference.new_zeros(())

    if difference_value < 0.0:
        spectral1 = augment_spectral_representation(
            spectral1, omega_max, -difference
        )
        spectral2 = augment_spectral_representation(
            spectral2, 0.0, zero_mass
        )
    else:
        spectral1 = augment_spectral_representation(
            spectral1, 0.0, zero_mass
        )
        spectral2 = augment_spectral_representation(
            spectral2, omega_max, difference
        )

    return spectral1, spectral2


def spectral_wasserstein2_squared(
    spectral1: Tensor,
    spectral2: Tensor,
    *,
    check_mass: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> Tensor:
    """
    Exact one-dimensional squared Wasserstein transport between two discrete
    spectral measures, using the merged-cumulative-endpoint algorithm used by
    current SPECTER.

    The inputs must be sorted by omega and have equal total mass.
    """
    for name, spectral in (("spectral1", spectral1), ("spectral2", spectral2)):
        if spectral.ndim != 2 or spectral.shape[1] != 2:
            raise ValueError(f"`{name}` must have shape (M, 2).")
        if torch.any(spectral[:, 1] < 0):
            raise ValueError(f"`{name}` contains negative masses.")

    if spectral1.device != spectral2.device:
        raise ValueError("Both spectra must be on the same device.")

    omega1 = spectral1[:, 0]
    mass1 = spectral1[:, 1]
    omega2 = spectral2[:, 0]
    mass2 = spectral2[:, 1]

    total1 = mass1.sum()
    total2 = mass2.sum()

    if check_mass and not torch.isclose(total1, total2, rtol=rtol, atol=atol):
        raise ValueError(
            "Balanced 1D transport requires equal total spectral mass. "
            f"Received {float(total1.detach().cpu()):.8g} and "
            f"{float(total2.detach().cpu()):.8g}."
        )

    endpoints = torch.cat((torch.cumsum(mass1, 0), torch.cumsum(mass2, 0)))
    masses = torch.cat((mass1, mass2))

    source = torch.cat(
        (
            torch.zeros(mass1.numel(), dtype=torch.long, device=mass1.device),
            torch.ones(mass2.numel(), dtype=torch.long, device=mass2.device),
        )
    )

    order = torch.argsort(endpoints)
    source_sorted = source[order]
    masses_sorted = masses[order]

    signed = torch.where(
        source_sorted == 0,
        masses_sorted,
        -masses_sorted,
    )
    running_difference = torch.cumsum(signed, dim=0)

    previous_source = torch.cat(
        (
            source_sorted.new_full((1,), -1),
            source_sorted[:-1],
        )
    )

    transported = torch.where(
        source_sorted == previous_source,
        masses_sorted,
        torch.where(
            source_sorted == 0,
            running_difference,
            -running_difference,
        ),
    )

    count1 = torch.cumsum((source_sorted == 0).to(torch.long), dim=0)
    count2 = torch.cumsum((source_sorted == 1).to(torch.long), dim=0)

    index1 = torch.where(
        source_sorted == 0,
        count1 - 1,
        count1,
    ).clamp(0, mass1.numel() - 1)

    index2 = torch.where(
        source_sorted == 1,
        count2 - 1,
        count2,
    ).clamp(0, mass2.numel() - 1)

    delta = omega1[index1] - omega2[index2]
    return torch.sum(transported * delta.square())
