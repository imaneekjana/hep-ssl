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
    def __init__(self, angle_range=(0, 2*np.pi)):
        self.angle_range = angle_range

    def __call__(self, event: CaloEvent):
        event_c = event.copy()
        
        theta = np.random.uniform(*self.angle_range)

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

        noise = np.abs(np.random.normal(
            loc=0.0,
            scale=self.sigma,
            size=len(event.hits)
        ))

        event_c.hits[:, 3] += noise

        if self.clip_min is not None:
            event_c.hits[:, 3] = np.clip(
                event_c.hits[:, 3],
                self.clip_min,
                None
            )

        return event_c

class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, event):
        for t in self.transforms:
            event = t(event)
        return event



