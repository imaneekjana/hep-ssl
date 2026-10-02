"""Bounded true-GravNet training, persistence and inductive-protocol checks."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import load_config, save_config
from src.prepare_pairwise import prepare
from src.data.events import load_prepared
from src.data.views import PairDataset, collate_pairs
from src.losses.multitask import MultiTaskObjective
from src.models.multispace import MultiSpaceEncoder
from src.training.checkpoint import assert_prepared_matches, load_checkpoint, metadata_fingerprint
from src.training.trainer import EventBatchSampler, build_optimizer, run_epoch, train


def small_config():
    cfg = load_config()
    cfg["data"]["events_per_channel"] = 10
    cfg["grid"].update(n_eta=8, n_phi=16)
    cfg["training"].update(epochs=2, batch_size=4, device="cpu")
    return cfg


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("data") / "prepared"
    return prepare(small_config(), path, synthetic=True)


def test_batching_including_singleton():
    assert list(EventBatchSampler(9, 4)) == [[0, 1, 2, 3], [4, 5, 6, 7, 8]]
    assert list(EventBatchSampler(3, 2)) == [[0, 1, 2]]
    assert list(EventBatchSampler(2, 32)) == [[0, 1]]
    assert list(EventBatchSampler(9, 4, drop_last=True)) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    with pytest.raises(ValueError):
        EventBatchSampler(1, 4)
    sampler = EventBatchSampler(12, 4, shuffle=True)
    first = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_optimizer_covers_every_parameter_once():
    model = MultiSpaceEncoder()
    objective = MultiTaskObjective()
    optimizer = build_optimizer(model, objective, small_config()["training"])
    actual = [p for g in optimizer.param_groups for p in g["params"]]
    expected = list(model.parameters()) + list(objective.parameters())
    assert len({id(p) for p in actual}) == len(actual) == len(expected)
    assert {id(p) for p in actual} == {id(p) for p in expected}
    assert optimizer.param_groups[1]["weight_decay"] == 0


def test_epoch_boundary_resume_is_identical(prepared, tmp_path):
    cfg = small_config()
    full = load_checkpoint(train(cfg, prepared.directory, tmp_path / "full"))
    partial_path = train(cfg, prepared.directory, tmp_path / "resumed", stop_after_epoch=1)
    partial = load_checkpoint(partial_path)
    restored = load_checkpoint(train(None, prepared.directory, tmp_path / "resumed", resume=partial_path))
    assert full["history"] == restored["history"]
    assert full["global_step"] == restored["global_step"] == 6
    assert full["scheduler_state"] == restored["scheduler_state"]
    for group in ("model_state", "objective_state"):
        for name, value in full[group].items():
            assert torch.equal(value, restored[group][name]), name
    for name, value in full["objective_state"].items():
        assert torch.any(value != partial["objective_state"][name])
    changed = copy.deepcopy(prepared.metadata)
    changed["feature_stats"]["mean"][0] += 1
    with pytest.raises(ValueError):
        assert_prepared_matches(full, changed)
    different = copy.deepcopy(full["config"])
    different["objective"]["tau"] *= 2
    with pytest.raises(ValueError):
        train(different, prepared.directory, tmp_path / "resumed", resume=partial_path)


@pytest.mark.parametrize("mode", ["single_cosine", "single_anisotropic", "five_anisotropic_no_aux"])
def test_other_modes_train(prepared, tmp_path, mode):
    cfg = small_config()
    cfg["mode"] = mode
    cfg["training"]["epochs"] = 1
    state = load_checkpoint(train(cfg, prepared.directory, tmp_path / mode))
    assert np.isfinite(state["history"][0]["val"]["loss"])
    assert state["history"][0]["val"]["physics"] == 0


def test_validation_preserves_training_stream(prepared):
    cfg = small_config()
    training = PairDataset(prepared, "train", cfg["augmentation"], seed=100)
    validation = PairDataset(prepared, "val", cfg["augmentation"], seed=200)
    training.set_epoch(1)
    before = training[0]
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(700)
    loader = DataLoader(validation, batch_sampler=EventBatchSampler(len(validation), 3),
                        collate_fn=collate_pairs, generator=generator)
    model, objective = MultiSpaceEncoder(), MultiTaskObjective()
    # Model construction legitimately consumed RNG; snapshot evaluation only.
    torch_state = torch.get_rng_state().clone()
    fingerprint = metadata_fingerprint(prepared.metadata)
    run_epoch(model, objective, loader, torch.device("cpu"))
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    after = training[0]
    assert torch.equal(before["view1"].x, after["view1"].x)
    assert metadata_fingerprint(prepared.metadata) == fingerprint
    validation.set_epoch(80)
    assert torch.equal(validation[0]["view1"].x, PairDataset(prepared, "val", cfg["augmentation"], seed=200)[0]["view1"].x)


def test_configuration_and_artifact_mismatch(prepared, tmp_path):
    cfg = small_config()
    cfg["grid"]["n_phi"] = 32
    with pytest.raises(ValueError, match="disagrees"):
        train(cfg, prepared.directory, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cli_help_has_no_execution_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for module in ("src.prepare_pairwise", "src.train_pairwise", "src.evaluate_pairwise"):
        result = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                                cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_worker_count_does_not_change_events_or_views(prepared):
    cfg = small_config()
    dataset = PairDataset(prepared, "train", cfg["augmentation"], seed=555)
    dataset.set_epoch(3)
    def collect(workers):
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=workers,
                            persistent_workers=False, collate_fn=collate_pairs,
                            generator=torch.Generator().manual_seed(1))
        return [(tuple(item["event_key"]), item["view1"].x.clone(), item["targets1"]["phi"].clone())
                for item in loader]
    single, multiple = collect(0), collect(2)
    assert len(single) == len(multiple)
    for one, two in zip(single, multiple):
        assert one[0] == two[0]
        assert torch.equal(one[1], two[1])
        assert torch.equal(one[2], two[2])
    assert len({key for batch in multiple for key in batch[0]}) == len(dataset)


def test_mutable_revision_and_invalid_stat_tolerance_rejected():
    from src.config import validate_config
    for revision in ("main", "master", "latest", ""):
        cfg = small_config()
        cfg["data"]["dataset_revision"] = revision
        with pytest.raises(ValueError):
            validate_config(cfg)
    cfg = small_config()
    cfg["targets"]["constant_tolerance"] = -1
    with pytest.raises(ValueError):
        validate_config(cfg)
