"""Stable raw-event identities, persisted splits and portable prepared inputs."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from src.data.projection import GridSpec


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts):
    return int(json_hash(parts)[:16], 16)


@dataclass
class RawEvent:
    hits: np.ndarray
    event_id: str
    channel: str
    dataset_revision: str
    pileup: str = "pu0"
    shard: str = ""

    def __post_init__(self):
        self.hits = np.asarray(self.hits, dtype=np.float64)
        if self.hits.ndim != 2 or self.hits.shape[1] != 4:
            raise ValueError("Raw hits must have columns x_mm,y_mm,z_mm,energy_GeV.")
        if not np.isfinite(self.hits[:, 3]).all() or np.any(self.hits[:, 3] < 0):
            raise ValueError("Raw energies must be finite nonnegative GeV.")
        if self.event_id is None:
            raise ValueError("event_id must not be null.")
        self.event_id = str(self.event_id)
        if not self.event_id or not self.dataset_revision or not self.channel:
            raise ValueError("Stable event_id, channel and dataset_revision are required.")

    @property
    def key(self):
        # JSON tuple encoding avoids delimiter collisions and preserves provenance.
        return json.dumps([self.dataset_revision, self.channel, self.pileup, self.shard, self.event_id], separators=(",", ":"))


def split_manifest(events, seed=42, fractions=(0.6, 0.2, 0.2), channels=None):
    if not np.allclose(fractions, (0.6, 0.2, 0.2), rtol=0, atol=1e-12):
        raise ValueError("This baseline fixes the channel-stratified split to 60/20/20.")
    keys = [event.key for event in events]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate stable event identities; supply actual shard identity if needed.")
    channels = list(channels or sorted({e.channel for e in events}))
    if len(channels) != 2 or len(set(channels)) != 2 or set(channels) != {e.channel for e in events}:
        raise ValueError("A pairwise run requires exactly two distinct configured channels.")
    result = []
    for label, channel in enumerate(channels):
        members = sorted((e for e in events if e.channel == channel), key=lambda e: (stable_seed(seed, e.key), e.key))
        n_train, n_val = int(len(members) * .6), int(len(members) * .2)
        if min(n_train, n_val, len(members) - n_train - n_val) < 1:
            raise ValueError("At least five events per channel are needed for a nonempty 60/20/20 split.")
        for index, event in enumerate(members):
            role = "train" if index < n_train else "val" if index < n_train + n_val else "test"
            result.append({"key": event.key, "event_id": event.event_id, "channel": channel,
                           "dataset_revision": event.dataset_revision, "pileup": event.pileup,
                           "shard": event.shard, "split": role, "label": label})
    return sorted(result, key=lambda row: row["key"])


def save_events(path, events):
    offsets = np.concatenate(([0], np.cumsum([len(e.hits) for e in events])))
    np.savez_compressed(path, hits=np.concatenate([e.hits for e in events]), offsets=offsets,
                        event_id=np.asarray([e.event_id for e in events]), channel=np.asarray([e.channel for e in events]),
                        dataset_revision=np.asarray([e.dataset_revision for e in events]),
                        pileup=np.asarray([e.pileup for e in events]), shard=np.asarray([e.shard for e in events]))


def load_events_npz(path, *, channel=None, dataset_revision=None, pileup=None):
    """NPZ contract: hits[sum N,4], offsets[M+1], event_id[M]; no pickle."""
    with np.load(path, allow_pickle=False) as data:
        hits, offsets, ids = data["hits"], data["offsets"], data["event_id"]
        if offsets.ndim != 1 or len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(hits) or np.any(np.diff(offsets) < 0) or not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("Invalid ragged hit offsets.")
        result = []
        for i, event_id in enumerate(ids):
            src_channel = str(data["channel"][i]) if "channel" in data else channel
            revision = str(data["dataset_revision"][i]) if "dataset_revision" in data else dataset_revision
            if channel is not None and src_channel != channel:
                raise ValueError("NPZ channel disagrees with --input channel.")
            if dataset_revision is not None and revision != dataset_revision:
                raise ValueError("NPZ revision disagrees with configured dataset revision.")
            source_pileup = str(data["pileup"][i]) if "pileup" in data else (pileup or "pu0")
            if pileup is not None and source_pileup != pileup:
                raise ValueError("NPZ pileup disagrees with configured pileup.")
            result.append(RawEvent(hits[offsets[i]:offsets[i+1]].copy(), str(event_id), src_channel, revision,
                                   source_pileup,
                                   str(data["shard"][i]) if "shard" in data else ""))
    return result


def load_local_channel(path, channel, dataset_revision, pileup="pu0", max_events=None):
    """Read explicit local files only: never starts a network download.

    Parquet must contain per-event list columns x,y,z,total_energy and event_id.
    Optional shard_id disambiguates upstream IDs; filenames/row numbers never do.
    """
    path = Path(path).expanduser().resolve()
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive.")
    if path.suffix == ".npz":
        return load_events_npz(path, channel=channel, dataset_revision=dataset_revision, pileup=pileup)[:max_events]
    import polars as pl
    paths = sorted(path.glob("**/*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No local parquet files found in {path}")
    events = []
    for source in paths:
        schema = pl.read_parquet_schema(source)
        required = {"event_id", "x", "y", "z", "total_energy"}
        if not required <= set(schema):
            raise ValueError(f"{source} lacks {sorted(required - set(schema))}; stable event_id is mandatory.")
        columns = sorted(required | ({"shard_id"} if "shard_id" in schema else set()))
        remaining = None if max_events is None else max_events - len(events)
        frame = pl.read_parquet(source, columns=columns, n_rows=remaining)
        for row in frame.iter_rows(named=True):
            if row["event_id"] is None:
                raise ValueError(f"Null event_id in {source}")
            hits = np.column_stack([row[name] for name in ("x", "y", "z", "total_energy")])
            events.append(RawEvent(hits, str(row["event_id"]), channel, dataset_revision, pileup, str(row.get("shard_id", ""))))
        if max_events is not None and len(events) >= max_events:
            break
    return events


@dataclass
class PreparedData:
    events: list
    metadata: dict
    directory: Path

    @property
    def manifest(self):
        return self.metadata["manifest"]

    @property
    def grid(self):
        return GridSpec.from_dict(self.metadata["grid"])

    @property
    def feature_stats(self):
        return self.metadata["feature_stats"]

    @property
    def summary_stats(self):
        return self.metadata["summary_stats"]

    @property
    def target_stats(self):
        return self.metadata["target_stats"]

    @property
    def target_definitions(self):
        return self.metadata["target_definitions"]


def load_prepared(path):
    path = Path(path).expanduser().resolve()
    directory = path.parent if path.is_file() else path
    with open(directory / "prepared.json") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != "prepared_v1":
        raise ValueError("Unsupported prepared schema.")
    check = dict(metadata)
    recorded = check.pop("metadata_hash")
    if json_hash(check) != recorded:
        raise ValueError("Prepared metadata hash mismatch.")
    for filename, expected in metadata["file_hashes"].items():
        if Path(filename).name != filename or file_hash(directory / filename) != expected:
            raise ValueError(f"Prepared input hash mismatch: {filename}")
    with open(directory / "manifest.json") as handle:
        manifest = json.load(handle)
    if manifest != metadata["manifest"] or json_hash(manifest) != metadata["manifest_hash"]:
        raise ValueError("Prepared manifest mismatch.")
    events = load_events_npz(directory / "events.npz")
    if [e.key for e in events] != [m["key"] for m in manifest]:
        raise ValueError("Prepared event identities/order do not match manifest.")
    return PreparedData(events, metadata, directory)
