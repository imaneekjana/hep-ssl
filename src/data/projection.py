from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    eta_edges: Tuple[float, ...]
    phi_edges: Tuple[float, ...]
    range_source: str
    cell_cutoff_gev: float = 0.0
    energy_ref_gev: float = 1.0
    log_epsilon_gev: float = 1e-6
    projection_version: str = "eta_phi_v1"
    axis_order: Tuple[str, str] = ("phi", "eta")
    position_unit: str = "mm"
    energy_unit: str = "GeV"

    def __post_init__(self):
        for name in ("eta_edges", "phi_edges"):
            edges = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                edges.ndim != 1
                or edges.size < 2
                or not np.isfinite(edges).all()
                or not np.all(np.diff(edges) > 0)
            ):
                raise ValueError(f"{name} must be finite, increasing edges.")
            object.__setattr__(self, name, tuple(edges.tolist()))

        object.__setattr__(self, "axis_order", tuple(self.axis_order))
        if self.phi_edges[0] != -np.pi or self.phi_edges[-1] != np.pi:
            raise ValueError("phi edges must span [-pi, pi].")
        if self.axis_order != ("phi", "eta"):
            raise ValueError("Only [n_phi, n_eta] axis order is supported.")
        if (
            self.projection_version != "eta_phi_v1"
            or self.position_unit != "mm"
            or self.energy_unit != "GeV"
        ):
            raise ValueError("Unsupported projection version or units.")
        if self.range_source not in {
            "explicit", "verified_geometry", "training_envelope"
        }:
            raise ValueError("Record how the eta range was determined.")

        values = np.array([
            self.cell_cutoff_gev,
            self.energy_ref_gev,
            self.log_epsilon_gev,
        ])
        if (
            not np.isfinite(values).all()
            or values[0] < 0
            or np.any(values[1:] <= 0)
        ):
            raise ValueError("Invalid cutoff, energy reference, or epsilon.")

    @classmethod
    def uniform(
        cls, eta_min, eta_max, *, n_eta=32, n_phi=32,
        range_source="explicit", **kwargs
    ):
        for size in (n_eta, n_phi):
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
                raise ValueError("Grid sizes must be positive integers.")
            if size < 1:
                raise ValueError("Grid sizes must be positive integers.")
        return cls(
            eta_edges=tuple(np.linspace(eta_min, eta_max, n_eta + 1)),
            phi_edges=tuple(np.linspace(-np.pi, np.pi, n_phi + 1)),
            range_source=range_source,
            **kwargs,
        )

    @property
    def shape(self):
        return len(self.phi_edges) - 1, len(self.eta_edges) - 1

    @property
    def eta_centers(self):
        edges = np.asarray(self.eta_edges)
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def phi_centers(self):
        edges = np.asarray(self.phi_edges)
        return 0.5 * (edges[:-1] + edges[1:])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass
class Projection:
    energy_grid: np.ndarray       # [n_phi, n_eta], linear GeV after cutoff
    node_features: np.ndarray     # [N, 3], eta, phi, log-energy; unstandardized
    node_energy: np.ndarray       # [N], linear GeV, separate from features
    diagnostics: dict

    @property
    def is_empty(self):
        return self.node_energy.size == 0


def project_hits(hits, grid: GridSpec) -> Projection:
    """Project [N, 4] x/y/z/E hits; eta endpoints are both accepted.

    Invalid/axis coordinates are dropped and counted. Energies must be
    finite and nonnegative. Cutoff is applied after cell energy summation.
    """
    hits = np.asarray(hits, dtype=np.float64)
    if hits.ndim != 2 or hits.shape[1] != 4:
        raise ValueError("Expected hits with shape [N, 4].")

    xyz, energy = hits[:, :3], hits[:, 3]
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("Hit energies must be finite, nonnegative GeV.")

    input_energy = float(energy.sum())
    if not np.isfinite(input_energy):
        raise ValueError("Total input energy overflowed.")

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        rho = np.hypot(xyz[:, 0], xyz[:, 1])
        eta = np.arcsinh(xyz[:, 2] / rho)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(rho)
        & (rho > 0)
        & np.isfinite(eta)
        & np.isfinite(phi)
    )
    outside = valid & (
        (eta < grid.eta_edges[0]) | (eta > grid.eta_edges[-1])
    )
    accepted = valid & ~outside

    # Passing phi first directly gives [n_phi, n_eta].
    summed, _, _ = np.histogram2d(
        phi[accepted],
        eta[accepted],
        bins=(grid.phi_edges, grid.eta_edges),
        weights=energy[accepted],
    )
    active = summed > grid.cell_cutoff_gev
    energy_grid = np.where(active, summed, 0.0)
    phi_index, eta_index = np.nonzero(active)
    node_energy = energy_grid[active].copy()
    log_energy = np.log(
        node_energy / grid.energy_ref_gev
        + grid.log_epsilon_gev / grid.energy_ref_gev
    )
    node_features = np.column_stack((
        grid.eta_centers[eta_index],
        grid.phi_centers[phi_index],
        log_energy,
    ))
    if not np.isfinite(node_features).all():
        raise ValueError("Nonfinite projected features.")

    invalid_energy = float(energy[~valid].sum())
    outside_energy = float(energy[outside].sum())
    cutoff_energy = float(summed[~active].sum())
    dropped_energy = invalid_energy + outside_energy + cutoff_energy
    diagnostics = {
        "n_input_hits": len(hits),
        "n_invalid_geometry_hits": int((~valid).sum()),
        "n_outside_eta_hits": int(outside.sum()),
        "n_active_cells": int(active.sum()),
        "input_energy_gev": input_energy,
        "invalid_geometry_energy_gev": invalid_energy,
        "outside_eta_energy_gev": outside_energy,
        "below_cutoff_energy_gev": cutoff_energy,
        "retained_energy_gev": float(node_energy.sum()),
        "dropped_energy_fraction": (
            dropped_energy / input_energy if input_energy > 0 else None
        ),
    }
    return Projection(energy_grid, node_features, node_energy, diagnostics)


def energy_summary(energy_grid, grid: GridSpec) -> np.ndarray:
    """Return [log(S/E_ref), log(S_T_dep/E_ref)] from this grid only."""
    energy_grid = np.asarray(energy_grid, dtype=np.float64)
    if (
        energy_grid.shape != grid.shape
        or not np.isfinite(energy_grid).all()
        or np.any(energy_grid < 0)
    ):
        raise ValueError("Expected a finite, nonnegative energy grid.")

    _, eta_index = np.nonzero(energy_grid > 0)
    energy = energy_grid[energy_grid > 0]
    if energy.size == 0:
        raise ValueError("An empty event has no log-energy summary.")

    total = float(energy.sum())
    if not np.isfinite(total):
        raise ValueError("Total grid energy overflowed.")

    eta = grid.eta_centers[eta_index]
    log_cosh = np.logaddexp(eta, -eta) - np.log(2.0)
    log_terms = np.log(energy) - log_cosh
    maximum = log_terms.max()
    log_transverse = maximum + np.log(np.exp(log_terms - maximum).sum())
    return np.array([
        np.log(total) - np.log(grid.energy_ref_gev),
        log_transverse - np.log(grid.energy_ref_gev),
    ])
