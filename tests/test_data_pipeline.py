"""Tests of energy, identities, independent references and train-only fitting."""
import copy
import json
import numpy as np
import pytest
from src.config import load_config
from src.data.events import RawEvent, load_prepared, split_manifest
from src.data.projection import GridSpec, energy_summary, project_hits
from src.prepare_pairwise import prepare, synthetic_events
from src.physics.targets import PhysicsTargets, fit_stats


def hits(eta, phi, energy):
    eta, phi, energy = np.broadcast_arrays(eta, phi, energy)
    return np.column_stack((1000 * np.cos(phi.ravel()), 1000 * np.sin(phi.ravel()), 1000 * np.sinh(eta.ravel()), energy.ravel()))


@pytest.fixture
def grid():
    return GridSpec.uniform(-2., 2., n_eta=8, n_phi=16)


@pytest.fixture
def small_config():
    cfg = load_config()
    cfg["data"]["events_per_channel"] = 10
    cfg["grid"].update(n_eta=8, n_phi=16)
    return cfg


def test_phi_targets(grid):
    target = PhysicsTargets(grid)
    e = np.zeros(grid.shape)
    e[0, 2], e[8, 2] = 1, 1
    raw = target(e)
    np.testing.assert_allclose(raw["phi"], [0, 1, 0, 1], atol=1e-14)
    assert raw["eta"].sum() == pytest.approx(1)
    assert (raw["eta"] >= 0).all()
    e[:, 2] = 1
    np.testing.assert_allclose(target(e)["phi"], 0, atol=1e-14)


def test_scale_and_rotation_invariance(grid):
    target = PhysicsTargets(grid)
    e = np.zeros(grid.shape)
    e[0, 0], e[7, 4], e[15, 5] = 2, 3, 4
    raw, scaled, rotated = target(e), target(3 * e), target(np.roll(e, 5, axis=0))
    np.testing.assert_allclose(scaled["energy"], raw["energy"] + np.log(3))
    for name in ("eta", "phi", "local"):
        np.testing.assert_allclose(scaled[name], raw[name], atol=1e-14)
    for name in raw:
        np.testing.assert_allclose(rotated[name], raw[name], atol=1e-14)
    assert np.all(np.diff(raw["local"]) >= -1e-14)


def test_cached_local_kernel_matches_periodic_direct_pairs(grid):
    target = PhysicsTargets(grid)
    assert target.kernels is PhysicsTargets(grid).kernels
    e = np.zeros(grid.shape)
    e[0, 1], e[15, 2] = 2, 3
    nodes = [(grid.eta_centers[1], grid.phi_centers[0], .4), (grid.eta_centers[2], grid.phi_centers[15], .6)]
    direct = []
    for ell in target.local_scales:
        total = 0
        for eta1, phi1, p1 in nodes:
            for eta2, phi2, p2 in nodes:
                dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
                total += p1 * p2 * np.exp(-((eta1 - eta2)**2 + dphi**2) / (2 * ell**2))
        direct.append(total)
    np.testing.assert_allclose(target(e)["local"], direct)


def test_target_constraints(grid):
    with pytest.raises(ValueError):
        PhysicsTargets(GridSpec.uniform(-2, 2, n_eta=8, n_phi=8))
    stats = fit_stats([[2, 1e-12], [2, 2e-12]])
    assert stats["constant_components"] == [True, True]
    assert stats["scale"] == [1, 1]


def test_split_is_stable_and_channels_disambiguate():
    events = synthetic_events(["ggf", "ttbar"], 10)
    manifest = split_manifest(events)
    assert manifest == split_manifest(events[::-1])
    assert len({r["key"] for r in manifest}) == 20
    for channel in ("ggf", "ttbar"):
        assert [sum(r["channel"] == channel and r["split"] == split for r in manifest) for split in ("train", "val", "test")] == [6, 2, 2]
    with pytest.raises(ValueError, match="Duplicate"):
        split_manifest(events + events[:1])


def test_fitting_only_training_identities_and_hash_validation(tmp_path, small_config):
    events = synthetic_events(small_config["data"]["channels"], 10)
    first = prepare(small_config, tmp_path / "a", events=events)
    changed = copy.deepcopy(events)
    holdout = {r["key"] for r in first.manifest if r["split"] != "train"}
    for event in changed:
        if event.key in holdout:
            event.hits[:, 2] *= 1000
            event.hits[:, 3] *= 100
    # Changed holdout may be partly outside range; retain a valid hit per event.
    for event in changed:
        if event.key in holdout:
            event.hits[0, :3] = [1000, 0, 0]
    second = prepare(small_config, tmp_path / "b", events=changed)
    for key in ("grid", "feature_stats", "summary_stats", "target_stats", "fit_event_keys"):
        assert first.metadata[key] == second.metadata[key]
    assert not set(first.metadata["fit_event_keys"]) & holdout
    assert load_prepared(tmp_path / "a").metadata == first.metadata
    with open(tmp_path / "a" / "events.npz", "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prepared(tmp_path / "a")


def test_view_reference_coordinates_and_energy(grid):
    from src.data.views import make_view
    sample = hits(np.linspace(-1, 1, 30), np.linspace(-2, 2, 30), np.linspace(1, 2, 30))
    config = {"order": ["energy_noise", "rotate", "xyz_noise", "crop", "shift"], "energy_noise": .5,
              "rotation": .8, "xyz_noise": 5, "crop_fraction": 1, "shift_std": 20}
    obs, ref, records = make_view(sample, config, np.random.default_rng(4))
    np.testing.assert_allclose(obs[:, :3], ref[:, :3])
    np.testing.assert_allclose(ref[:, 3], sample[:, 3])
    assert not np.array_equal(obs[:, 3], ref[:, 3])
    assert [p["name"] for p in records] == config["order"]
    assert not np.shares_memory(obs, ref)


def test_no_augmentation_identity():
    from src.data.views import make_view
    sample = hits([-.5, .5], [.2, -.2], [1, 2])
    for config in ({"order": []}, {"order": ["energy_noise", "rotate", "xyz_noise", "crop", "shift"]}):
        observed, reference, _ = make_view(sample, config, np.random.default_rng(1))
        np.testing.assert_array_equal(observed, sample)
        np.testing.assert_array_equal(reference, sample)


def test_datasets_rng_and_target_isolation(tmp_path, small_config):
    import torch
    from src.data.views import PairDataset, CleanDataset, collate_pairs, collate_clean
    prepared = prepare(small_config, tmp_path / "prepared", synthetic=True)
    train = PairDataset(prepared, "train", small_config["augmentation"], seed=4)
    val = PairDataset(prepared, "val", small_config["augmentation"], seed=4)
    train.set_epoch(3)
    before = train[0]
    val_before = val[0]
    val.set_epoch(100)
    val_after = val[0]
    after = train[0]
    assert torch.equal(before["view1"].x, after["view1"].x)
    assert torch.equal(val_before["view1"].x, val_after["view1"].x)
    assert torch.equal(before["view1"].energy, after["view1"].energy)
    train.set_epoch(4)
    assert not torch.equal(before["view1"].x, train[0]["view1"].x)
    batch = collate_pairs([train[0], train[1]])
    assert batch["view1"].summary.shape == (2, 2)
    assert batch["targets1"]["eta"].shape == (2, 8)
    assert batch["view1"].num_graphs == 2
    clean = CleanDataset(prepared)
    item = clean[0]
    features = item["graph"].x.clone()
    item["targets"]["energy"].fill_(999)
    assert torch.equal(item["graph"].x, features)
    assert all(not t.requires_grad for t in item["targets"].values())
    assert not hasattr(item["graph"], "targets")
    assert collate_clean([clean[0], clean[1]])["graph"].summary.shape == (2, 2)
    pure = PairDataset(prepared, "train", {"order": []})
    clean_train = CleanDataset(prepared, "train")
    assert torch.equal(pure[0]["view1"].x, clean_train[0]["graph"].x)
    assert torch.equal(pure[0]["targets1"]["local"], clean_train[0]["targets"]["local"])


def test_local_npz_and_parquet_inputs(tmp_path, small_config):
    import polars as pl
    from src.data.events import load_events_npz, load_local_channel, save_events
    events = synthetic_events(small_config["data"]["channels"], 10)
    inputs = {}
    for channel in small_config["data"]["channels"]:
        members = [e for e in events if e.channel == channel]
        npz = tmp_path / f"{channel}.npz"
        save_events(npz, members)
        loaded = load_events_npz(npz)
        assert [e.key for e in loaded] == [e.key for e in members]
        np.testing.assert_array_equal(loaded[0].hits, members[0].hits)
        records = [{"event_id": e.event_id, "x": e.hits[:, 0].tolist(), "y": e.hits[:, 1].tolist(),
                    "z": e.hits[:, 2].tolist(), "total_energy": e.hits[:, 3].tolist()} for e in members]
        parquet = tmp_path / f"{channel}.parquet"
        pl.DataFrame(records).write_parquet(parquet)
        adapter = load_local_channel(parquet, channel, "synthetic-v1")
        assert [e.key for e in adapter] == [e.key for e in members]
        np.testing.assert_allclose(adapter[0].hits, members[0].hits)
        inputs[channel] = str(parquet)
    small_config["data"]["dataset_revision"] = "synthetic-v1"
    prepared = prepare(small_config, tmp_path / "local", inputs=inputs)
    assert len(prepared.events) == 20
    assert prepared.metadata["data"]["dataset_revision"] == "synthetic-v1"


def test_reuse_manifest_and_empty_event_accounting(tmp_path, small_config):
    events = synthetic_events(small_config["data"]["channels"], 10)
    events[0].hits[:, 3] = 0
    first = prepare(small_config, tmp_path / "first", events=events)
    assert first.metadata["excluded_event_count"] == 1
    row = next(r for r in first.manifest if r["key"] == events[0].key)
    assert not row["usable"] and row["exclusion_reason"] == "zero_accepted_energy"
    second = prepare(small_config, tmp_path / "second", events=events[::-1], manifest_path=tmp_path / "first" / "manifest.json")
    assert second.manifest == first.manifest
    assert second.metadata["split_config"]["reused"]
    assert second.metadata["feature_stats"] == first.metadata["feature_stats"]


def test_two_geometry_views_have_independent_targets(tmp_path, small_config):
    from src.data.views import PairDataset
    prepared = prepare(small_config, tmp_path / "prepared", synthetic=True)
    dataset = PairDataset(prepared, "train", {"order": ["shift"], "shift_std": [0, 0, 300]}, seed=7)
    example = dataset[0]
    # Energy references are unpolluted, but changed geometric eta coordinates
    # lead to view-specific targets; copying targets1 to targets2 would fail.
    assert not np.allclose(example["targets1"]["eta"].numpy(), example["targets2"]["eta"].numpy())


def test_preparation_preserves_existing_files_and_checks_identity(tmp_path, small_config):
    events = synthetic_events(small_config["data"]["channels"], 10)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    existing = occupied / "events.npz"
    existing.write_bytes(b"existing-result")
    with pytest.raises(FileExistsError):
        prepare(small_config, occupied, events=events)
    assert existing.read_bytes() == b"existing-result"
    cfg = copy.deepcopy(small_config)
    cfg["data"]["dataset_revision"] = "wrong-revision"
    with pytest.raises(ValueError, match="revision"):
        prepare(cfg, tmp_path / "bad_revision", events=events)
    cfg = copy.deepcopy(small_config)
    cfg["data"]["pileup"] = "pu-other"
    with pytest.raises(ValueError, match="pileup"):
        prepare(cfg, tmp_path / "bad_pileup", events=events)
    manifest = split_manifest(events)
    manifest[0]["shard"] = "forged-shard"
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest identity"):
        prepare(small_config, tmp_path / "bad_manifest", events=events, manifest_path=path)


def test_npz_pileup_and_bounded_parquet_loader(tmp_path):
    import polars as pl
    from src.data.events import load_events_npz, load_local_channel, save_events
    events = synthetic_events(["ggf"], 10)
    archive = tmp_path / "raw.npz"
    save_events(archive, events)
    with pytest.raises(ValueError, match="pileup"):
        load_events_npz(archive, pileup="pu-other")
    records = [{"event_id": e.event_id, "x": e.hits[:, 0].tolist(), "y": e.hits[:, 1].tolist(),
                "z": e.hits[:, 2].tolist(), "total_energy": e.hits[:, 3].tolist(), "unused_truth": [1, 2, 3]} for e in events]
    source = tmp_path / "parts"
    source.mkdir()
    pl.DataFrame(records[:5]).write_parquet(source / "01.parquet")
    # A later unreadable shard proves bounded selection stops after its limit.
    (source / "02.parquet").write_bytes(b"not-needed")
    selected = load_local_channel(source, "ggf", "synthetic-v1", max_events=3)
    assert [e.event_id for e in selected] == [e.event_id for e in events[:3]]
