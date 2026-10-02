"""Real GravNet extraction tests; no dataset download or long training."""

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.evaluate_pairwise import build_frozen_encoder, default_output_dir
from src.evaluation.representations import export_representations, load_representations
from src.models.multispace import MultiSpaceEncoder


def sample_checkpoint():
    config = {"mode": "five_anisotropic_physics", "model": {
        "hidden_dim": 4, "latent_dim": 64, "proj_dim": 32,
        "k": 2, "space_dim": 2, "propagate_dim": 4,
    }}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(21)
        model = MultiSpaceEncoder(config["mode"], **config["model"])
    return {"config": config, "model_state": model.state_dict()}


def clean_batch():
    rng = np.random.default_rng(61)
    graphs = [Data(x=torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32),
                   energy=torch.tensor(rng.uniform(0.1, 2, size=7), dtype=torch.float32),
                   summary=torch.zeros(1, 2)) for _ in range(3)]
    targets = {"energy": torch.zeros(3, 2), "eta": torch.full((3, 8), 1 / 8),
               "phi": torch.zeros(3, 4), "local": torch.full((3, 4), 0.5)}
    return {"graph": Batch.from_data_list(graphs), "targets": targets,
            "raw_targets": {key: value.clone() for key, value in targets.items()},
            "event_key": ["v1/a/1", "v1/b/2", "v1/a/3"],
            "label": torch.tensor([0, 1, 0]), "split": ["train", "val", "test"]}


def test_pretrained_random_share_constructor_and_preserve_rng():
    checkpoint = sample_checkpoint()
    before = torch.get_rng_state().clone()
    pretrained = build_frozen_encoder(checkpoint, encoder_mode="pretrained", random_seed=4)
    random = build_frozen_encoder(checkpoint, encoder_mode="random", random_seed=5)
    duplicate = build_frozen_encoder({"config": checkpoint["config"]}, encoder_mode="random", random_seed=5)
    assert torch.equal(before, torch.get_rng_state())
    assert list(pretrained.state_dict()) == list(random.state_dict())
    assert all(not parameter.requires_grad for parameter in random.parameters())
    assert not random.training
    for name, value in pretrained.state_dict().items():
        assert torch.equal(value, checkpoint["model_state"][name])
    for name, value in random.state_dict().items():
        assert torch.equal(value, duplicate.state_dict()[name])
    assert any(not torch.equal(value, random.state_dict()[name]) for name, value in pretrained.state_dict().items())


def test_real_gravnet_clean_export_shapes_order_and_target_separation(tmp_path):
    model = build_frozen_encoder(sample_checkpoint())
    batch = clean_batch()
    path = tmp_path / "representations.npz"
    arrays = export_representations(model, [batch], path, metadata={"test": True})
    assert arrays["h_concat"].shape == (3, 320)
    for name in ("general", "energy", "eta", "phi", "local"):
        assert arrays[f"h_{name}"].shape == (3, 64)
        assert arrays[f"z_{name}"].shape == (3, 32)
    for task, dim in (("energy", 2), ("eta", 8), ("phi", 4), ("local", 4)):
        assert arrays[f"prediction_{task}"].shape == (3, dim)
        assert arrays[f"target_{task}"].shape == (3, dim)
    assert arrays["baseline_energy_activity"].shape == (3, 3)
    assert arrays["baseline_physics_summaries"].shape == (3, 19)
    assert arrays["n_active_cells"].tolist() == [7, 7, 7]
    loaded = load_representations(path)
    np.testing.assert_array_equal(loaded["event_key"], batch["event_key"])
    # Neither the target dictionaries nor process labels enter graph.forward.
    batch["targets"]["energy"] += 100
    batch["raw_targets"]["energy"] -= 50
    batch["label"] = 1 - batch["label"]
    changed = export_representations(model, [batch])
    np.testing.assert_array_equal(arrays["h_concat"], changed["h_concat"])


def test_default_classifier_directory_follows_experiment_stage_layout(tmp_path):
    run = tmp_path / "09_28 training" / "pretraining" / "example"
    assert default_output_dir(run, "pretrained") == tmp_path / "09_28 training" / "classifier" / "example" / "pretrained"
    assert default_output_dir(tmp_path / "custom", "random_42") == tmp_path / "custom" / "evaluation" / "random_42"
