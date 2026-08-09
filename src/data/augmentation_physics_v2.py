import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os

#os.chdir('/global/cfs/cdirs/m4474/aneek/particlemind')

#from src.datasets.CLDHits import CLDHits #CLDHits is a data processing module for processing .parquet files


'''
An object representing calorimeter hits

'''

import numpy as np


class CaloEvent:
    """
    Represents a calorimeter event with hits:
    columns: [x, y, z, E]
    """

    def __init__(self, hits: np.ndarray):
        assert hits.ndim == 2 and hits.shape[1] == 4
        self.hits = hits.astype(np.float32)

    @property
    def xyz(self):
        return self.hits[:, :3]

    @property
    def energy(self):
        return self.hits[:, 3]

    def copy(self):
        return CaloEvent(self.hits.copy())

    def apply(self, transform):
        return transform(self)

    def __repr__(self):
        return f"CaloEvent(num_hits={len(self.hits)})"


class Transform:
    def __call__(self, event: CaloEvent) -> CaloEvent:
        raise NotImplementedError


class RandomRotateXY(Transform):
    def __init__(self, angle_range=(0, 2*np.pi), gaussian = False):
        self.angle_range = angle_range

        self.gaussian = gaussian

    def __call__(self, event: CaloEvent):
        event_c = event.copy()

        if self.gaussian == False:

            theta = np.random.uniform(*self.angle_range)

        else:

            theta = np.random.normal(loc=0.0, scale=self.angle_range[1])
        
        

        #theta = np.pi/2

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x = event.hits[:, 0]
        y = event.hits[:, 1]

        event_c.hits[:, 0] = cos_t * x - sin_t * y
        event_c.hits[:, 1] = sin_t * x + cos_t * y

        return event_c
        
class RandomShift(Transform):
    def __init__(self, shift_std=(1.0, 1.0, 0.0)):
        self.shift_std = shift_std

    def __call__(self, event: CaloEvent):
        event = event.copy()
        dx = np.random.normal(0, self.shift_std[0])
        dy = np.random.normal(0, self.shift_std[1])
        dz = np.random.normal(0, self.shift_std[2])

        event.hits[:, 0] += dx
        event.hits[:, 1] += dy
        event.hits[:, 2] += dz

        return event


class RandomBeamReflectionZ(Transform):
    """Exchange the two beam directions by reflecting the event across z=0."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, event: CaloEvent):
        event_c = event.copy()

        if np.random.random() < self.probability:
            event_c.hits[:, 2] *= -1.0

        return event_c


class RandomSpatialCrop(Transform):
    """
    Masks hits inside a random spatial box by setting their energy to zero.
    Array length remains unchanged.
    """

    def __init__(self, crop_fraction=0.2):
        self.crop_fraction = crop_fraction

    def __call__(self, event: CaloEvent):
        event_c = event.copy()

        xyz = event.xyz
        N = len(xyz)

        if N == 0:
            return event

        # Choose random hit as center
        center = xyz[np.random.randint(N)]

        # Estimate scale of box from spread
        spread = np.std(xyz, axis=0)
        radius = self.crop_fraction * spread

        # Identify hits inside box
        inside_mask = np.all(
            np.abs(xyz - center) < radius,
            axis=1
        )

        # Zero out energies instead of deleting hits
        event_c.hits[inside_mask, 3] = 0.0

        return event_c


class EnergyWhiteNoise(Transform):
    """
    Adds Gaussian white noise to calorimeter hit energies.

    E -> E + N(0, sigma)

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian noise.
    clip_min : float or None
        If set, energies are clipped below this value (e.g., 0.0).
    """

    def __init__(self, sigma=0.1, clip_min=0.0):
        self.sigma = sigma
        self.clip_min = clip_min

    def __call__(self, event: CaloEvent):
        event_c = event.copy()

        event_c.hits = event.hits.copy()

        noise = self.sigma*np.random.normal(
            loc=0.0,
            scale=1.0,
            size=event_c.hits.shape[0])
        

        log_en = event_c.hits[:, 3]

        event_c.hits[:, 3] = log_en + noise


        return event_c

class NoiseXYZ(Transform):

    def __init__(self, sigma=5.0, clip_min=0.0):
        self.sigma = sigma
        self.clip_min = clip_min

    def __call__(self, event: CaloEvent):
        
        event_c = event.copy()

        event_c.hits = event.hits.copy()

        for i in range(3):

            noise = self.sigma*np.random.normal(
            loc=0.0,
            scale=1.0,
            size=event_c.hits.shape[0])
        

            x_i = event_c.hits[:, i]

            event_c.hits[:, i] = x_i + noise
        

        return event_c


class GridTransform:
    """Base class for augmentations acting on a linear cell-energy grid."""

    def __call__(self, grid):
        raise NotImplementedError


class RandomCyclicPhiRoll(GridTransform):
    """
    Apply a uniformly sampled element of the cyclic group C_n along phi.

    The eta-phi grid is stored as [phi_bin, eta_bin], so phi_axis=0.
    """

    def __init__(self, phi_axis=0):
        self.phi_axis = phi_axis

    def __call__(self, grid):
        grid_c = np.asarray(grid).copy()
        n_phi = grid_c.shape[self.phi_axis]
        shift = np.random.randint(0, n_phi)
        return np.roll(grid_c, shift=shift, axis=self.phi_axis)


class SoftCellMask(GridTransform):
    """
    Remove only soft cells, with a strict event-level energy budget.

    A cell is eligible when E_cell / E_event is below
    ``cell_fraction_threshold``. The total removed energy is at most a
    uniformly sampled fraction in [0, max_removed_fraction].
    """

    def __init__(self, max_removed_fraction=0.01,
                 cell_fraction_threshold=1e-3):
        self.max_removed_fraction = max_removed_fraction
        self.cell_fraction_threshold = cell_fraction_threshold

    def __call__(self, grid):
        grid_c = np.asarray(grid, dtype=float).copy()
        total_energy = float(grid_c.sum())

        if total_energy <= 0.0 or self.max_removed_fraction <= 0.0:
            return grid_c

        flat = grid_c.reshape(-1)
        fractions = flat / total_energy
        candidates = np.flatnonzero(
            (flat > 0.0) &
            (fractions < self.cell_fraction_threshold)
        )
        np.random.shuffle(candidates)

        budget = np.random.uniform(0.0, self.max_removed_fraction) * total_energy
        removed = 0.0

        for index in candidates:
            cell_energy = float(flat[index])
            if removed + cell_energy <= budget:
                flat[index] = 0.0
                removed += cell_energy

        return grid_c


class LocalCellEnergySharing(GridTransform):
    """
    Redistribute a limited energy budget between immediately adjacent cells.

    Energy is conserved exactly. Phi neighbors are periodic, while eta does
    not wrap. The eta-phi grid convention is [phi_bin, eta_bin].
    """

    def __init__(self, max_moved_fraction=0.02,
                 transfer_fraction_range=(0.1, 0.3)):
        self.max_moved_fraction = max_moved_fraction
        self.transfer_fraction_range = transfer_fraction_range

    def __call__(self, grid):
        grid_c = np.asarray(grid, dtype=float).copy()
        total_energy = float(grid_c.sum())

        if total_energy <= 0.0 or self.max_moved_fraction <= 0.0:
            return grid_c

        budget = np.random.uniform(0.0, self.max_moved_fraction) * total_energy
        moved_total = 0.0
        n_phi, n_eta = grid_c.shape
        sources = np.argwhere(grid_c > 0.0)
        np.random.shuffle(sources)

        for phi_index, eta_index in sources:
            remaining_budget = budget - moved_total
            if remaining_budget <= 0.0:
                break

            neighbors = [
                ((phi_index - 1) % n_phi, eta_index),
                ((phi_index + 1) % n_phi, eta_index),
            ]
            if eta_index > 0:
                neighbors.append((phi_index, eta_index - 1))
            if eta_index + 1 < n_eta:
                neighbors.append((phi_index, eta_index + 1))

            target_phi, target_eta = neighbors[
                np.random.randint(0, len(neighbors))
            ]
            alpha = np.random.uniform(*self.transfer_fraction_range)
            moved = min(
                alpha * float(grid_c[phi_index, eta_index]),
                remaining_budget,
            )

            grid_c[phi_index, eta_index] -= moved
            grid_c[target_phi, target_eta] += moved
            moved_total += moved

        return grid_c


class ComposeGrid(GridTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, grid):
        grid_c = np.asarray(grid).copy()
        for transform in self.transforms:
            grid_c = transform(grid_c)
        return grid_c




class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, event):
        event_c = event.copy()
        for t in self.transforms:
            event_c = t(event_c)
        return event_c


