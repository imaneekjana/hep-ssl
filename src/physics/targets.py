"""Scheme B targets computed only from clean reference energy grids."""
from functools import lru_cache
import numpy as np
from src.data.projection import GridSpec, energy_summary


@lru_cache(maxsize=8)
def _geometry(eta_edges, phi_edges, scales):
    eta = (np.asarray(eta_edges[:-1]) + eta_edges[1:]) / 2
    phi = (np.asarray(phi_edges[:-1]) + phi_edges[1:]) / 2
    pp, ee = np.meshgrid(phi, eta, indexing="ij")
    deta = ee.ravel()[:, None] - ee.ravel()[None, :]
    delta = pp.ravel()[:, None] - pp.ravel()[None, :]
    dphi = np.arctan2(np.sin(delta), np.cos(delta))
    kernels = np.stack([np.exp(-(deta**2 + dphi**2) / (2 * ell**2)) for ell in scales])
    kernels.setflags(write=False)
    return kernels


class PhysicsTargets:
    def __init__(self, grid: GridSpec, local_scales=None, eta_regions=8, phi_orders=(1, 2, 3, 4)):
        self.grid = grid
        if eta_regions != 8 or grid.shape[1] % eta_regions:
            raise ValueError("Eight eta regions require n_eta divisible by 8.")
        if tuple(phi_orders) != (1, 2, 3, 4) or max(phi_orders) >= grid.shape[0] / 2:
            raise ValueError("Phi orders 1..4 must be strictly below grid Nyquist.")
        if local_scales is None:
            ell = max(np.max(np.diff(grid.eta_edges)), np.max(np.diff(grid.phi_edges)))
            local_scales = [ell, 2 * ell, 4 * ell, 8 * ell]
        self.local_scales = tuple(float(x) for x in local_scales)
        if len(self.local_scales) != 4 or not np.isfinite(self.local_scales).all() or np.any(np.asarray(self.local_scales) <= 0):
            raise ValueError("Four finite positive local scales are required.")
        self.eta_regions = eta_regions
        self.phi_orders = tuple(phi_orders)
        self.kernels = _geometry(grid.eta_edges, grid.phi_edges, self.local_scales)
        self.phase = np.exp(1j * np.asarray(phi_orders)[:, None] * grid.phi_centers[None, :])

    @property
    def definitions(self):
        step = self.grid.shape[1] // self.eta_regions
        return {"version": "scheme_b_v1", "eta_regions": self.eta_regions,
                "eta_region_edges": list(self.grid.eta_edges[::step]),
                "phi_orders": list(self.phi_orders), "local_scales": list(self.local_scales),
                "local_includes_self_pairs": True, "units": "eta_phi_direction"}

    def __call__(self, energy_grid):
        energy = np.asarray(energy_grid, dtype=np.float64)
        t_energy = energy_summary(energy, self.grid)
        p = energy / energy.sum()
        eta = p.sum(axis=0).reshape(self.eta_regions, -1).sum(axis=1)
        q = self.phase @ p.sum(axis=1)
        phi = q.real**2 + q.imag**2
        active = np.flatnonzero(p)
        weights = p.ravel()[active]
        local = np.array([weights @ kernel[np.ix_(active, active)] @ weights for kernel in self.kernels])
        return {"energy": t_energy, "eta": eta, "phi": phi, "local": local}


def fit_stats(values, atol=1e-8):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Stats require a finite, nonempty matrix.")
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    constant = std <= atol * np.maximum(1.0, np.abs(mean))
    scale = np.where(constant, 1.0, std)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "std": std.tolist(),
            "constant_components": constant.tolist(), "count": len(values),
            "weighting": "rows", "ddof": 0, "constant_atol": atol}


def standardize(values, stats):
    return (np.asarray(values, dtype=np.float64) - np.asarray(stats["mean"])) / np.asarray(stats["scale"])


def inverse_standardize(values, stats):
    return np.asarray(values) * np.asarray(stats["scale"]) + np.asarray(stats["mean"])


def standardize_targets(targets, stats):
    return {name: value.copy() if name == "eta" else standardize(value, stats[name]) for name, value in targets.items()}
