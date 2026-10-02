"""Two deterministic views with matched geometry and separate energy references."""
import numpy as np
from torch.utils.data import Dataset
from src.data.events import stable_seed
from src.data.projection import project_hits, energy_summary
from src.physics.targets import PhysicsTargets, standardize, standardize_targets


TRANSFORMS = {"rotate", "energy_noise", "xyz_noise", "shift", "crop"}


def make_view(hits, augmentation, rng):
    """Return observed/reference copies and actual sampled parameters.

    Positions are mm; energy noise is additive independent GeV Gaussian followed
    by clipping at zero. Crop zeros a box of radius fraction*per-axis std around
    an observed hit. Shift is one common XY Gaussian offset (z=0). Neither shift
    nor jitter is claimed to be a collision symmetry. Order is preserved.
    """
    observed = np.asarray(hits, dtype=np.float64).copy()
    reference = observed.copy()
    order = augmentation.get("order", [])
    if isinstance(order, str):
        order = [] if order == "none" else order.split("+")
    if any(name not in TRANSFORMS for name in order):
        raise ValueError("Unknown augmentation in explicit order.")
    records = []
    for name in order:
        record = {"name": name}
        if name == "rotate":
            scale = float(augmentation.get("rotation", 0.0))
            mode = augmentation.get("rotation_mode", "uniform")
            if scale < 0 or not np.isfinite(scale) or mode not in {"uniform", "gaussian"}:
                raise ValueError("Invalid rotation range/distribution.")
            angle = rng.uniform(-scale, scale) if mode == "uniform" else rng.normal(0, scale)
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            for array in (observed, reference):
                array[:, :2] = array[:, :2] @ rotation.T
            record["angle_rad"] = float(angle)
        elif name in {"xyz_noise", "shift"}:
            scale = np.asarray(augmentation.get("xyz_noise" if name == "xyz_noise" else "shift_std", 0.0), dtype=float)
            if not np.isfinite(scale).all() or np.any(scale < 0):
                raise ValueError("Position standard deviation must be finite nonnegative mm.")
            if name == "xyz_noise":
                delta = rng.normal(size=(len(observed), 3)) * scale
            else:
                std = np.array([float(scale), float(scale), 0.0]) if scale.ndim == 0 else scale
                if std.shape != (3,):
                    raise ValueError("shift_std must be scalar XY std or a three-element XYZ std.")
                delta = rng.normal(size=3) * std
            observed[:, :3] += delta
            reference[:, :3] += delta
            record["offset_mm"] = delta.copy()
        elif name == "energy_noise":
            scale = float(augmentation.get("energy_noise", 0.0))
            if scale < 0 or not np.isfinite(scale):
                raise ValueError("Energy noise must be finite nonnegative GeV.")
            delta = rng.normal(0, scale, len(observed))
            observed[:, 3] = np.maximum(0.0, observed[:, 3] + delta)
            record["offset_gev"] = delta
        else:
            fraction = float(augmentation.get("crop_fraction", 0.0))
            if fraction < 0 or not np.isfinite(fraction):
                raise ValueError("Crop fraction must be finite and nonnegative.")
            valid = np.flatnonzero(np.isfinite(observed[:, :3]).all(axis=1))
            mask = np.zeros(len(observed), dtype=bool)
            if len(valid):
                center = observed[rng.choice(valid), :3].copy()
                radius = fraction * observed[valid, :3].std(axis=0)
                mask = np.all(np.abs(observed[:, :3] - center) < radius, axis=1)
                record.update(center_mm=center, radius_mm=radius)
            observed[mask, 3] = 0.0
            record["mask"] = mask
        records.append(record)
    return observed, reference, records


def _target_builder(prepared):
    definition = prepared.target_definitions
    return PhysicsTargets(prepared.grid, local_scales=definition["local_scales"],
                          eta_regions=definition["eta_regions"], phi_orders=definition["phi_orders"])


def graph_from_projection(projected, prepared):
    import torch
    from torch_geometric.data import Data
    if projected.is_empty:
        raise ValueError("Augmentation produced zero accepted energy; reduce strength or adjust configured acceptance. No fake nodes or hidden retries are used.")
    return Data(x=torch.tensor(standardize(projected.node_features, prepared.feature_stats), dtype=torch.float32),
                energy=torch.tensor(projected.node_energy.copy(), dtype=torch.float32),
                summary=torch.tensor(standardize(energy_summary(projected.energy_grid, prepared.grid), prepared.summary_stats)[None, :], dtype=torch.float32))


def _tensor_targets(values):
    import torch
    # Independent copies and no autograd path from target to input.
    return {k: torch.tensor(np.array(v, copy=True), dtype=torch.float32) for k, v in values.items()}


class PairDataset(Dataset):
    def __init__(self, prepared, split, augmentation, seed=42):
        if split not in {"train", "val", "test"}:
            raise ValueError("Choose a manifest split.")
        self.prepared, self.split, self.augmentation, self.seed = prepared, split, augmentation, seed
        self.indices = [i for i, row in enumerate(prepared.manifest) if row["split"] == split and row["usable"]]
        self.epoch = 0
        self.targets = _target_builder(prepared)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        position = self.indices[index]
        event, row = self.prepared.events[position], self.prepared.manifest[position]
        result = {"event_key": event.key, "label": row["label"]}
        # Validation/test streams do not depend on training epoch or global RNG.
        epoch_key = self.epoch if self.split == "train" else "fixed_validation"
        for view_id in (1, 2):
            rng = np.random.default_rng(stable_seed(self.seed, epoch_key, event.key, view_id))
            observed, reference, _ = make_view(event.hits, self.augmentation, rng)
            obs_projection = project_hits(observed, self.prepared.grid)
            ref_projection = project_hits(reference, self.prepared.grid)
            if ref_projection.is_empty:
                raise ValueError(f"Reference has zero accepted energy for event {event.key}, view {view_id}.")
            result[f"view{view_id}"] = graph_from_projection(obs_projection, self.prepared)
            result[f"targets{view_id}"] = _tensor_targets(standardize_targets(self.targets(ref_projection.energy_grid), self.prepared.target_stats))
        return result


class CleanDataset(Dataset):
    def __init__(self, prepared, split=None):
        if split is not None and split not in {"train", "val", "test"}:
            raise ValueError("Choose a manifest split or None.")
        self.prepared = prepared
        self.indices = [i for i, row in enumerate(prepared.manifest) if row["usable"] and (split is None or row["split"] == split)]
        self.targets = _target_builder(prepared)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        position = self.indices[index]
        event, row = self.prepared.events[position], self.prepared.manifest[position]
        projected = project_hits(event.hits, self.prepared.grid)
        raw_targets = self.targets(projected.energy_grid)
        return {"graph": graph_from_projection(projected, self.prepared),
                "targets": _tensor_targets(standardize_targets(raw_targets, self.prepared.target_stats)),
                "raw_targets": _tensor_targets(raw_targets), "event_key": event.key,
                "label": row["label"], "split": row["split"]}


def _collate_targets(items, field):
    import torch
    return {key: torch.stack([item[field][key] for item in items]) for key in items[0][field]}


def collate_pairs(items):
    import torch
    from torch_geometric.data import Batch
    return {"view1": Batch.from_data_list([x["view1"] for x in items]),
            "view2": Batch.from_data_list([x["view2"] for x in items]),
            "targets1": _collate_targets(items, "targets1"), "targets2": _collate_targets(items, "targets2"),
            "event_key": [x["event_key"] for x in items], "label": torch.tensor([x["label"] for x in items])}


def collate_clean(items):
    import torch
    from torch_geometric.data import Batch
    return {"graph": Batch.from_data_list([x["graph"] for x in items]),
            "targets": _collate_targets(items, "targets"), "raw_targets": _collate_targets(items, "raw_targets"),
            "event_key": [x["event_key"] for x in items], "label": torch.tensor([x["label"] for x in items]),
            "split": [x["split"] for x in items]}
