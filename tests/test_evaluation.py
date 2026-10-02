"""Meaningful checks for the inductive downstream protocol."""

import numpy as np
import pytest

from src.evaluation.representations import (
    _eta_objective, _softmax, cross_head_diagnostics, evaluate_representations,
    fit_physics_probe, load_representations, physics_metrics,
    predict_physics_probe, representation_diagnostics, validate_representations,
)


def make_export():
    rng = np.random.default_rng(28)
    n = 48
    x = rng.normal(size=(n, 4))
    # Held-out rows have deliberately different means to expose leakage.
    x[32:] += 3
    role = np.asarray(["train"] * 32 + ["val"] * 8 + ["test"] * 8)
    labels = np.tile([0, 1], n // 2)
    x[:, 0] += labels
    target_energy = np.column_stack((x[:, 0], x[:, 1]))
    arrays = {
        "event_key": np.asarray([f"revision/channel/event_{i}" for i in range(n)]),
        "split": role, "label": labels,
        "h_general": x, "h_concat": x.copy(),
        "target_energy": target_energy,
        "target_eta": _softmax(x @ rng.normal(size=(4, 8)) / 3),
        "target_phi": _softmax(x),
        "target_local": np.sort(_softmax(x), axis=1),
        "baseline_energy_activity": target_energy,
    }
    return arrays


def test_export_refuses_duplicate_identity_and_reordered_concat():
    arrays = make_export()
    validate_representations(arrays)
    arrays["event_key"][1] = arrays["event_key"][0]
    with pytest.raises(ValueError, match="unique"):
        validate_representations(arrays)
    arrays = make_export()
    arrays["h_concat"] = arrays["h_concat"][:, ::-1]
    with pytest.raises(ValueError, match="fixed representation space order"):
        validate_representations(arrays)


def test_eta_distribution_objective_has_correct_gradient():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(5, 3))
    target = _softmax(rng.normal(size=(5, 4)))
    parameters = rng.normal(size=16)
    value, grad = _eta_objective(parameters, x, target, 0.7)
    numeric = []
    for i in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[i] = 1e-6
        plus = _eta_objective(parameters + delta, x, target, 0.7)[0]
        minus = _eta_objective(parameters - delta, x, target, 0.7)[0]
        numeric.append((plus - minus) / 2e-6)
    assert np.isfinite(value)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)


def test_eta_probe_never_standardizes_distribution_targets():
    arrays = make_export()
    mask = arrays["split"] == "train"
    probe = fit_physics_probe(arrays["h_general"][mask], arrays["target_eta"][mask], "eta")
    assert "target_scaler" not in probe
    np.testing.assert_allclose(probe["x_scaler"].mean_, arrays["h_general"][mask].mean(axis=0))
    prediction = predict_physics_probe(probe, arrays["h_general"])
    np.testing.assert_allclose(prediction.sum(axis=1), 1)
    assert np.all(prediction > 0)
    assert physics_metrics(prediction[mask], arrays["target_eta"][mask], "eta")["kl_target_prediction"] < 0.1


def test_complete_protocol_preserves_split_and_train_only_scalers(tmp_path):
    arrays = make_export()
    mask = arrays["split"] == "train"
    target_stats = {}
    for task in ("energy", "phi", "local"):
        mean = arrays[f"target_{task}"][mask].mean(axis=0)
        scale = arrays[f"target_{task}"][mask].std(axis=0)
        target_stats[task] = {"mean": mean, "scale": scale}
        arrays[f"prediction_{task}"] = (arrays[f"target_{task}"] - mean) / scale
    arrays["prediction_eta"] = np.log(arrays["target_eta"])
    result = evaluate_representations(arrays, tmp_path, target_stats=target_stats)
    assert set(result["metrics"]["classification"]) == {"general", "concat"}
    np.testing.assert_array_equal(result["predictions"]["split"], arrays["split"])
    for name, model in result["models"].items():
        if name.startswith("classification"):
            np.testing.assert_allclose(model[0].mean_, arrays["h_general"][mask].mean(axis=0))
        elif name.startswith("physics_probes"):
            np.testing.assert_allclose(model["x_scaler"].mean_, arrays["h_general"][mask].mean(axis=0))
            if model["kind"] == "linear_ridge":
                task = name.rsplit("/", 1)[1]
                np.testing.assert_allclose(model["target_scaler"].mean_, arrays[f"target_{task}"][mask].mean(axis=0))
    for task in ("energy", "eta", "phi", "local"):
        assert result["metrics"]["network_readouts"][task]["test"]["mse"] < 1e-25
    assert (tmp_path / "metrics.json").is_file()
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as predictions:
        np.testing.assert_array_equal(predictions["event_key"], arrays["event_key"])


def test_physics_metrics_do_not_clip_out_of_range_readout():
    result = physics_metrics(np.asarray([[-1., 2.]]), np.asarray([[0., 1.]]), "phi")
    assert result["mse"] == 1
    assert result["out_of_bounds_fraction"] == 1


def test_diagnostics_detect_collapse_and_redundancy():
    assert representation_diagnostics(np.ones((8, 4)))["effective_rank"] == 0
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 4))
    result = cross_head_diagnostics({"a": x, "b": x.copy()})
    assert result["a:b"]["linear_cka"] == pytest.approx(1)
    assert result["a:b"]["max_absolute_correlation"] == pytest.approx(1)


def test_representation_archive_needs_no_pickle(tmp_path):
    arrays = make_export()
    path = tmp_path / "representations.npz"
    np.savez_compressed(path, **arrays)
    loaded = load_representations(path)
    np.testing.assert_array_equal(loaded["event_key"], arrays["event_key"])


def test_no_new_split_created_when_manifest_incomplete():
    arrays = make_export()
    arrays["split"][arrays["split"] == "test"] = "train"
    with pytest.raises(ValueError, match="nonempty saved"):
        evaluate_representations(arrays)


def test_changed_test_inputs_and_targets_cannot_change_fitted_models():
    original = make_export()
    changed = {name: value.copy() for name, value in original.items()}
    heldout = changed["split"] == "test"
    for name in ("h_general", "h_concat", "target_energy", "baseline_energy_activity"):
        changed[name][heldout] += 1000
    changed["target_phi"][heldout] = 0.95
    changed["target_local"][heldout] = 0.05
    changed["target_eta"][heldout] = 1 / 8
    first = evaluate_representations(original)
    second = evaluate_representations(changed)
    for name, before in first["models"].items():
        after = second["models"][name]
        if name.startswith(("classification", "baselines")):
            np.testing.assert_array_equal(before[0].mean_, after[0].mean_)
            np.testing.assert_array_equal(before[1].coef_, after[1].coef_)
        else:
            np.testing.assert_array_equal(before["x_scaler"].mean_, after["x_scaler"].mean_)
            if before["kind"] == "linear_softmax_kl":
                np.testing.assert_array_equal(before["weight"], after["weight"])
            else:
                np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)
                np.testing.assert_array_equal(before["target_scaler"].mean_, after["target_scaler"].mean_)


def test_regression_probe_keeps_near_constant_components_unscaled():
    x = np.arange(32).reshape(8, 4)
    targets = np.column_stack((np.linspace(0, 1e-10, 8), np.arange(8)))
    probe = fit_physics_probe(x, targets, "energy")
    assert probe["target_scaler"].constant_components_.tolist() == [True, False]
    assert probe["target_scaler"].scale_[0] == 1
