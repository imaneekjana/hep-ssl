"""Prepare frozen train-only preprocessing and a shared inductive split."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from src.data.events import (RawEvent, PreparedData, file_hash, json_hash, load_local_channel,
                             load_prepared, save_events, split_manifest, stable_seed)
from src.data.projection import GridSpec, project_hits
from src.physics.targets import PhysicsTargets, fit_stats


def synthetic_events(channels, count, seed=42):
    """Synthetic geometry is an integration fixture, never ColliderML data."""
    events = []
    for label, channel in enumerate(channels):
        for i in range(count):
            rng = np.random.default_rng(stable_seed("synthetic-v1", seed, channel, i))
            n = 96
            eta = rng.uniform(-2, 2, n)
            phi = rng.uniform(-np.pi, np.pi, n)
            radius = rng.uniform(1000, 2000, n)
            energy = rng.lognormal(mean=.1 * label, sigma=.7, size=n)
            hits = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), radius * np.sinh(eta), energy))
            events.append(RawEvent(hits, f"synthetic-{i:06}", channel, "synthetic-v1"))
    return events


def _grid_from_training(events, manifest, settings):
    settings = dict(settings)
    eta_min, eta_max = settings.pop("eta_min", None), settings.pop("eta_max", None)
    if (eta_min is None) != (eta_max is None):
        raise ValueError("Specify both eta_min and eta_max, or neither.")
    source = "explicit"
    if eta_min is None:
        train_keys = {r["key"] for r in manifest if r["split"] == "train"}
        bound = 0.0
        for event in events:
            if event.key not in train_keys:
                continue
            x, y, z = event.hits[:, :3].T
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                eta = np.arcsinh(z / np.hypot(x, y))
            valid = np.isfinite(eta) & np.isfinite(event.hits[:, :3]).all(axis=1)
            if valid.any():
                bound = max(bound, float(np.abs(eta[valid]).max()))
        if bound <= 0:
            raise ValueError("Cannot infer nondegenerate eta range from clean training geometry; configure it explicitly.")
        bound += max(1.0, bound) * 1e-9
        eta_min, eta_max, source = -bound, bound, "training_envelope"
    return GridSpec.uniform(eta_min, eta_max, range_source=source, **settings)


def prepare(config, output_dir, *, events=None, inputs=None, synthetic=False, manifest_path=None):
    from src.config import validate_config
    config = validate_config(copy.deepcopy(config))
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Preparation output must be absent or empty: {output_dir}")
    data = config["data"]
    channels = data["channels"]
    if len(channels) != 2 or len(set(channels)) != 2:
        raise ValueError("Exactly two different channels are required.")
    if events is None:
        if synthetic:
            events = synthetic_events(channels, data["events_per_channel"], data.get("split_seed", 42))
            data["dataset_revision"] = "synthetic-v1"
        else:
            if not data.get("dataset_revision"):
                raise ValueError("Set a verified dataset_revision for real input; no revision is inferred from a mutable branch name.")
            if set(inputs or {}) != set(channels):
                raise ValueError("Supply one --input CHANNEL=LOCAL_PATH for each configured channel.")
            events = []
            for channel in channels:
                loaded = load_local_channel(inputs[channel], channel, data["dataset_revision"], data.get("pileup", "pu0"), max_events=data["events_per_channel"])
                ordered = sorted(loaded, key=lambda e: e.key)
                if len({e.key for e in ordered}) != len(ordered):
                    raise ValueError("Duplicate upstream IDs; include stable shard_id, not row numbers.")
                count = data["events_per_channel"]
                if len(ordered) < count:
                    raise ValueError(f"Requested {count} {channel} events; only {len(ordered)} are available locally.")
                events.extend(ordered[:count])
    events = list(events)
    revisions = {event.dataset_revision for event in events}
    if len(revisions) != 1:
        raise ValueError("A comparison cohort must use one explicit dataset revision.")
    revision = next(iter(revisions))
    if data.get("dataset_revision") not in (None, revision):
        raise ValueError("Raw-event revision differs from configured revision.")
    data["dataset_revision"] = revision
    if any(event.pileup != data.get("pileup", "pu0") for event in events):
        raise ValueError("Raw-event pileup differs from configured pileup.")
    proposed = split_manifest(events, data.get("split_seed", 42), data.get("split_fractions", [.6, .2, .2]), channels)
    if manifest_path is not None:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if {r["key"] for r in manifest} != {r["key"] for r in proposed} or len(manifest) != len(proposed):
            raise ValueError("Existing manifest must contain each selected stable event exactly once.")
        expected = {r["key"]: r for r in proposed}
        for row in manifest:
            identity = expected[row["key"]]
            if row["split"] not in ("train", "val", "test") or any(row[k] != identity[k] for k in ("label", "channel", "event_id", "dataset_revision", "pileup", "shard")):
                raise ValueError("Invalid existing manifest identity/label/split.")
        manifest = [{k: row[k] for k in expected[row["key"]]} for row in manifest]
    else:
        manifest = proposed
    by_key = {e.key: e for e in events}
    events = [by_key[row["key"]] for row in manifest]
    grid = _grid_from_training(events, manifest, config["grid"])
    target_config = dict(config.get("targets", {}))
    tolerance = target_config.pop("constant_tolerance", 1e-8)
    targets = PhysicsTargets(grid, **target_config)
    config["grid"]["eta_min"] = grid.eta_edges[0]
    config["grid"]["eta_max"] = grid.eta_edges[-1]
    config["targets"]["local_scales"] = list(targets.local_scales)
    train_features, train_targets = [], {k: [] for k in ("energy", "phi", "local")}
    for event, row in zip(events, manifest):
        projected = project_hits(event.hits, grid)
        row["usable"] = not projected.is_empty
        row["exclusion_reason"] = None if row["usable"] else "zero_accepted_energy"
        row["projection_diagnostics"] = projected.diagnostics
        if row["split"] == "train" and row["usable"]:
            train_features.append(projected.node_features)
            raw_targets = targets(projected.energy_grid)
            for name in train_targets:
                train_targets[name].append(raw_targets[name])
    for role in ("train", "val", "test"):
        if sum(r["usable"] and r["split"] == role for r in manifest) < 2:
            raise ValueError(f"At least two usable events are required in {role}; inspect empty-event diagnostics.")
    feature_stats = fit_stats(np.concatenate(train_features), tolerance)
    feature_stats["weighting"] = "clean_training_nodes"
    target_stats = {k: fit_stats(v, tolerance) for k, v in train_targets.items()}
    summary_stats = copy.deepcopy(target_stats["energy"])
    metadata = {
        "schema_version": "prepared_v1", "synthetic": bool(synthetic), "data": data,
        "grid": grid.to_dict(), "feature_stats": feature_stats, "summary_stats": summary_stats,
        "target_stats": target_stats, "target_definitions": targets.definitions,
        "manifest": manifest, "manifest_hash": json_hash(manifest),
        "split_config": {"seed": data.get("split_seed", 42), "fractions": [.6, .2, .2], "stratify": "channel", "reused": manifest_path is not None},
        "fit_event_keys": [r["key"] for r in manifest if r["split"] == "train" and r["usable"]],
        "excluded_event_count": sum(not r["usable"] for r in manifest),
        "preparation_config": config,
    }
    output_dir = Path(output_dir).expanduser().resolve()
    if (output_dir / "prepared.json").exists():
        raise FileExistsError(f"Prepared artifacts already exist: {output_dir}. Reuse them or choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_events(output_dir / "events.npz", events)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    metadata["file_hashes"] = {name: file_hash(output_dir / name) for name in ("events.npz", "manifest.json")}
    metadata["metadata_hash"] = json_hash(metadata)
    (output_dir / "prepared.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return load_prepared(output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input", action="append", default=[], metavar="CHANNEL=PATH")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--events-per-channel", type=int)
    parser.add_argument("--synthetic", action="store_true", help="Generate explicitly synthetic integration fixtures.")
    parser.add_argument("--manifest", help="Reuse an existing manifest with exactly these event identities.")
    args = parser.parse_args(argv)
    from src.config import load_config, validate_config
    config = load_config(args.config)
    if args.events_per_channel is not None:
        config["data"]["events_per_channel"] = args.events_per_channel
    if args.dataset_revision is not None:
        config["data"]["dataset_revision"] = args.dataset_revision
    validate_config(config)
    inputs = {}
    for item in args.input:
        channel, separator, path = item.partition("=")
        if not separator or not channel or not path or channel in inputs:
            parser.error("--input requires a unique CHANNEL=PATH.")
        inputs[channel] = path
    prepared = prepare(config, args.output_dir, inputs=inputs, synthetic=args.synthetic, manifest_path=args.manifest)
    print(json.dumps({"prepared_dir": str(prepared.directory), "synthetic": prepared.metadata["synthetic"],
                      "events": len(prepared.events), "excluded": prepared.metadata["excluded_event_count"],
                      "manifest_hash": prepared.metadata["manifest_hash"]}, indent=2))


if __name__ == "__main__":
    main()
