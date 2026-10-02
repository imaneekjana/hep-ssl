"""Small, explicit JSON configuration shared by preparation, training and evaluation."""
import copy
import json
import math
from pathlib import Path

MODES = ("single_cosine", "single_anisotropic", "five_anisotropic_no_aux", "five_anisotropic_physics")
DEFAULTS = {
    "mode": "five_anisotropic_physics",
    "prepared_dir": None,
    "data": {
        "dataset_id": "CERN/ColliderML-Release-1", "dataset_revision": None,
        "channels": ["ggf", "ttbar"], "pileup": "pu0", "events_per_channel": 2500,
        "split_seed": 42, "split_fractions": [0.6, 0.2, 0.2],
    },
    "grid": {
        "n_eta": 32, "n_phi": 32, "eta_min": None, "eta_max": None,
        "cell_cutoff_gev": 0.0, "energy_ref_gev": 1.0, "log_epsilon_gev": 1e-6,
    },
    "augmentation": {
        "order": ["energy_noise", "rotate", "crop"], "rotation": math.pi / 8,
        "rotation_mode": "uniform", "energy_noise": 1e-4, "xyz_noise": 5.0,
        "shift_std": 0.0, "crop_fraction": 0.5,
    },
    "targets": {
        "eta_regions": 8, "phi_orders": [1, 2, 3, 4], "local_scales": None,
        "constant_tolerance": 1e-8,
    },
    "model": {
        "hidden_dim": 16, "latent_dim": 64, "proj_dim": 32, "k": 8,
        "space_dim": 4, "propagate_dim": 16,
    },
    "objective": {"tau": 0.07, "gamma": 1.0},
    "training": {
        "epochs": 18, "batch_size": 32, "lr": 0.0003, "weight_decay": 0.0001,
        "seed": 42, "augmentation_seed": 142, "validation_seed": 242,
        "num_workers": 0, "device": "auto", "amp": False,
    },
}


def _merge(base, update, prefix=""):
    if not isinstance(update, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = copy.deepcopy(value)
    return base


def validate_config(config):
    if config["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    channels = config["data"]["channels"]
    if len(channels) != 2 or len(set(channels)) != 2 or not set(channels) <= {"ggf", "ttbar", "dihiggs"}:
        raise ValueError("Exactly two different supported channels are required")
    revision = config["data"]["dataset_revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip() or revision.lower() in {"main", "master", "latest"}):
        raise ValueError("dataset_revision must be an explicit frozen version, not a mutable branch")
    fractions = config["data"]["split_fractions"]
    if len(fractions) != 3 or any(not math.isfinite(v) or v <= 0 for v in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("split_fractions must contain three positive fractions summing to one")
    for section, names in {
        "data": ("events_per_channel",), "grid": ("n_eta", "n_phi"),
        "model": ("hidden_dim", "latent_dim", "proj_dim", "k", "space_dim", "propagate_dim"),
        "training": ("epochs", "batch_size"),
    }.items():
        for name in names:
            value = config[section][name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
    if config["training"]["batch_size"] < 2:
        raise ValueError("Contrastive batch_size must be >= 2")
    workers = config["training"]["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if config["model"]["latent_dim"] != 64 or config["model"]["proj_dim"] != 32:
        raise ValueError("Phase 1 requires 64-D representations and 32-D projections")
    order = config["augmentation"]["order"]
    if not isinstance(order, list) or len(order) != len(set(order)) or not set(order) <= {"rotate", "energy_noise", "xyz_noise", "shift", "crop"}:
        raise ValueError("augmentation.order must be an ordered list of distinct supported transforms")
    if config["augmentation"]["rotation_mode"] not in {"uniform", "gaussian"}:
        raise ValueError("rotation_mode must be uniform or gaussian")
    for name in ("rotation", "energy_noise", "xyz_noise", "shift_std", "crop_fraction"):
        value = config["augmentation"][name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"augmentation.{name} must be finite and nonnegative")
    for section, name, strictly_positive in (
        ("objective", "tau", True), ("objective", "gamma", False),
        ("training", "lr", True), ("training", "weight_decay", False),
    ):
        value = config[section][name]
        if not math.isfinite(value) or value < 0 or (strictly_positive and value == 0):
            raise ValueError(f"Invalid {section}.{name}")
    if config["targets"]["eta_regions"] != 8 or config["targets"]["phi_orders"] != [1, 2, 3, 4]:
        raise ValueError("Phase 1 target dimensions are fixed: eta=8 and phi orders=1..4")
    if config["grid"]["n_eta"] % 8 or config["grid"]["n_phi"] <= 8:
        raise ValueError("n_eta must be divisible by 8; n_phi must exceed 8 (Nyquist)")
    tolerance = config["targets"]["constant_tolerance"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("constant_tolerance must be finite and positive")
    return config


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            _merge(config, json.load(stream))
    return validate_config(config)


def save_config(config, path):
    Path(path).write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
