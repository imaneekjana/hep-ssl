"""Versioned training state; preprocessing must match exactly on restore."""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 1


def metadata_fingerprint(metadata):
    content = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def assert_prepared_matches(checkpoint, metadata):
    expected = checkpoint["prepared_fingerprint"]
    if metadata_fingerprint(checkpoint["preprocessing"]) != expected:
        raise ValueError("Checkpoint preprocessing fingerprint is inconsistent")
    if metadata_fingerprint(metadata) != expected:
        raise ValueError("Prepared data, manifest or statistics differ from checkpoint")


def capture_rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly resume CUDA RNG state on a CPU-only host")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def source_state():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    try:
        commit, dirty = git("rev-parse", "HEAD"), git("status", "--short")
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, "Git metadata unavailable"
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"source_commit": commit, "dirty_state": dirty, "source_sha256": digest.hexdigest()}


def save_checkpoint(payload, path):
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, map_location="cpu"):
    # Training states contain Python/NumPy RNG state. Load only your own run artifacts.
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or legacy checkpoint; implicit migration is disabled")
    required = {
        "config", "model_state", "objective_state", "optimizer_state", "scheduler_state",
        "scaler_state", "epoch", "global_step", "best_validation", "history",
        "rng_state", "preprocessing", "prepared_fingerprint",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"Incomplete checkpoint: {sorted(missing)}")
    assert_prepared_matches(state, state["preprocessing"])
    return state
