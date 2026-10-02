"""Clean representation export and fixed-manifest downstream evaluation.

All probe hyperparameters are fixed before looking at validation or test data.
Every feature/target scaler is fitted on the manifest's training events only.
The eta probe is a linear-logit softmax model fitted to distribution labels by
KL, with exactly the same protocol for every representation space.
"""

import json
from pathlib import Path

import numpy as np


SPACE_ORDER = ("general", "energy", "eta", "phi", "local")
TASK_ORDER = ("energy", "eta", "phi", "local")
SPLITS = ("train", "val", "test")


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_representations(arrays):
    """Reject ambiguous identities and misaligned export rows without resplitting."""
    ids = np.asarray(arrays["event_key"])
    split = np.asarray(arrays["split"])
    labels = np.asarray(arrays["label"])
    n = len(ids)
    if ids.ndim != 1 or n == 0 or len(set(ids.tolist())) != n:
        raise ValueError("Representation export requires unique nonempty event keys.")
    if split.shape != (n,) or not set(split.tolist()).issubset(SPLITS):
        raise ValueError("Unknown or misaligned manifest split roles.")
    if labels.shape != (n,):
        raise ValueError("Labels and event identities are misaligned.")
    heads = [name for name in SPACE_ORDER if f"h_{name}" in arrays]
    if not heads:
        raise ValueError("No encoder representations found.")
    for name, value in arrays.items():
        if name in {"metadata_json", "baseline_names_json"}:
            continue
        value = np.asarray(value)
        if value.ndim == 0 or len(value) != n:
            raise ValueError(f"Export field {name} has an inconsistent event dimension.")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Export field {name} contains nonfinite values.")
        if name.startswith(("h_", "z_", "target_", "prediction_", "training_target_", "baseline_")) and value.ndim != 2:
            raise ValueError(f"Export field {name} must be a matrix.")
    expected = np.concatenate([arrays[f"h_{name}"] for name in heads], axis=1)
    if "h_concat" not in arrays or not np.array_equal(arrays["h_concat"], expected):
        raise ValueError("h_concat must follow the fixed representation space order.")
    if "target_eta" in arrays:
        eta = np.asarray(arrays["target_eta"])
        if np.any(eta < 0) or not np.allclose(eta.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Eta labels must remain probability distributions.")
    return heads


def export_representations(model, loader, output_path=None, *, device="cpu", metadata=None):
    """Use the shared CleanDataset/collate_clean path for both encoder modes.

    Targets and labels stay outside the graph passed to the model. Predictions
    are stored separately from h; regression readouts retain training units.
    """
    import torch

    chunks = {}

    def add(name, value):
        chunks.setdefault(name, []).append(_as_numpy(value))

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            graph = batch["graph"].to(device)
            output = model(graph)
            heads = [name for name in SPACE_ORDER if name in output["h"]]
            for name in heads:
                add(f"h_{name}", output["h"][name])
                if name in output.get("z", {}):
                    add(f"z_{name}", output["z"][name])
            add("h_concat", torch.cat([output["h"][name] for name in heads], dim=1))
            for task, value in output.get("pred", {}).items():
                add(f"prediction_{task}", value)
            for task, value in batch["raw_targets"].items():
                add(f"target_{task}", value)
            for task, value in batch["targets"].items():
                add(f"training_target_{task}", value)
            add("event_key", np.asarray(batch["event_key"], dtype=str))
            add("split", np.asarray(batch["split"], dtype=str))
            add("label", batch["label"])
            add("n_active_cells", graph.ptr[1:] - graph.ptr[:-1])

    if not chunks:
        raise ValueError("Clean evaluation dataset contains no events.")
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    energy_activity = np.column_stack((
        arrays["target_energy"], np.log1p(arrays["n_active_cells"]),
    ))
    arrays["baseline_energy_activity"] = energy_activity
    arrays["baseline_physics_summaries"] = np.column_stack((
        energy_activity, *(arrays[f"target_{task}"] for task in ("eta", "phi", "local")),
    ))
    arrays["baseline_names_json"] = np.asarray(json.dumps({
        "energy_activity": ["log_total_deposited_energy", "log_transverse_deposited_energy", "log1p_active_cells"],
        "physics_summaries": "energy_activity followed by eta, phi, local targets in saved component order",
    }))
    arrays["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    validate_representations(arrays)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
    return arrays


def load_representations(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_representations(arrays)
    return arrays


def representation_diagnostics(features):
    """Centered spectrum: effective rank is entropy of normalized singular values."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite representation matrix.")
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = singular.sum()
    if total > 0:
        p = singular[singular > 0] / total
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        effective_rank = 0.0
    eigen = singular ** 2 / max(len(x) - 1, 1)
    participation = float(eigen.sum() ** 2 / np.sum(eigen ** 2)) if np.any(eigen) else 0.0
    variance = centered.var(axis=0)
    return {
        "n_events": len(x), "dimension": x.shape[1],
        "variance": variance.tolist(), "singular_values": singular.tolist(),
        "covariance_eigenvalues": eigen.tolist(),
        "effective_rank": effective_rank,
        "covariance_participation_ratio": participation,
        "near_constant_fraction": float(np.mean(variance <= 1e-12)),
    }


def cross_head_diagnostics(heads):
    """Linear CKA and component correlations; no independence constraint is fitted."""
    result = {}
    names = list(heads)
    for i, left in enumerate(names):
        x = np.asarray(heads[left], dtype=np.float64)
        x = x - x.mean(axis=0)
        for right in names[i + 1:]:
            y = np.asarray(heads[right], dtype=np.float64)
            y = y - y.mean(axis=0)
            cross = x.T @ y
            cka_denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
            norms = np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0))
            corr = np.divide(cross, norms, out=np.zeros_like(cross), where=norms > 0)
            result[f"{left}:{right}"] = {
                "linear_cka": float(np.sum(cross ** 2) / cka_denominator) if cka_denominator > 0 else None,
                "mean_absolute_correlation": float(np.abs(corr).mean()),
                "max_absolute_correlation": float(np.abs(corr).max()),
                "correlation": corr.tolist(),
            }
    return result


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def physics_metrics(prediction, target, task):
    """Regression uses physical target units; eta receives predicted probabilities."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    error = prediction - target
    result = {"mse": float(np.mean(error ** 2)), "mae": float(np.mean(np.abs(error))),
              "component_mse": np.mean(error ** 2, axis=0).tolist()}
    if task == "eta":
        log_target = np.log(np.maximum(target, np.finfo(float).tiny))
        log_pred = np.log(np.maximum(prediction, np.finfo(float).tiny))
        result["kl_target_prediction"] = float(np.mean(np.sum(target * (log_target - log_pred), axis=1)))
    else:
        denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
        numerator = np.sum(error ** 2, axis=0)
        result["component_r2"] = [float(1 - a / b) if b > 0 else None for a, b in zip(numerator, denominator)]
        if task in {"phi", "local"}:
            result["out_of_bounds_fraction"] = float(np.mean((prediction < 0) | (prediction > 1)))
    return result


def _eta_objective(parameters, x, target, alpha):
    """Mean KL up to target entropy + alpha/(2N) times squared linear weights."""
    from scipy.special import logsumexp

    dimensions, outputs = x.shape[1], target.shape[1]
    weight = parameters[:dimensions * outputs].reshape(dimensions, outputs)
    bias = parameters[dimensions * outputs:]
    logits = x @ weight + bias
    log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
    residual = (np.exp(log_probability) - target) / len(x)
    value = -np.mean(np.sum(target * log_probability, axis=1)) + alpha * np.sum(weight ** 2) / (2 * len(x))
    gradient = np.concatenate(((x.T @ residual + alpha * weight / len(x)).ravel(), residual.sum(axis=0)))
    return float(value), gradient


def fit_physics_probe(x_train, target_train, task, *, alpha=1.0, max_iter=1000):
    """Fit only the passed training rows. Fixed regularization is shared by spaces."""
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray(x_train, dtype=np.float64)
    target_train = np.asarray(target_train, dtype=np.float64)
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    if task == "eta":
        from scipy.optimize import minimize

        if np.any(target_train < 0) or not np.allclose(target_train.sum(axis=1), 1):
            raise ValueError("Eta probe targets must be unstandardized probability distributions.")
        dimensions, outputs = x_scaled.shape[1], target_train.shape[1]
        initial = np.zeros((dimensions + 1) * outputs)
        fitted = minimize(_eta_objective, initial, args=(x_scaled, target_train, alpha),
                          method="L-BFGS-B", jac=True, options={"maxiter": max_iter, "ftol": 1e-12})
        if not fitted.success:
            raise RuntimeError(f"Eta linear probe failed to converge: {fitted.message}")
        return {"kind": "linear_softmax_kl", "x_scaler": scaler,
                "weight": fitted.x[:dimensions * outputs].reshape(dimensions, outputs),
                "bias": fitted.x[dimensions * outputs:], "alpha": alpha,
                "iterations": int(fitted.nit)}
    from sklearn.linear_model import Ridge

    target_scaler = StandardScaler().fit(target_train)
    # Match preparation's treatment of near-constant physical components.
    # Persist this decision with the fitted probe rather than amplifying noise.
    target_scaler.constant_tolerance_ = 1e-8
    target_scaler.constant_components_ = (
        np.sqrt(target_scaler.var_) <= 1e-8 * np.maximum(1.0, np.abs(target_scaler.mean_))
    )
    target_scaler.scale_[target_scaler.constant_components_] = 1.0
    model = Ridge(alpha=alpha).fit(x_scaled, target_scaler.transform(target_train))
    return {"kind": "linear_ridge", "x_scaler": scaler,
            "target_scaler": target_scaler, "model": model, "alpha": alpha}


def predict_physics_probe(probe, features):
    x = probe["x_scaler"].transform(features)
    if probe["kind"] == "linear_softmax_kl":
        return _softmax(x @ probe["weight"] + probe["bias"])
    return probe["target_scaler"].inverse_transform(probe["model"].predict(x))


def evaluate_representations(arrays, output_dir=None, *, classifier_c=1.0, probe_alpha=1.0,
                             seed=42, target_stats=None):
    """Run all spaces and concat on the persisted train/val/test roles.

    Test predictions are produced only after each estimator has been fitted.
    There is no test-driven head selection or hyperparameter search here.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    heads = validate_representations(arrays)
    if classifier_c <= 0 or probe_alpha < 0:
        raise ValueError("Classifier C must be positive and probe alpha nonnegative.")
    roles, labels = np.asarray(arrays["split"]), np.asarray(arrays["label"])
    masks = {role: roles == role for role in SPLITS}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Evaluation requires nonempty saved train, val, and test partitions.")
    if any(not np.array_equal(np.unique(labels[mask]), [0, 1]) for mask in masks.values()):
        raise ValueError("Pairwise evaluation requires both labels 0 and 1 in every split.")
    destination = Path(output_dir) if output_dir is not None else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)

    def save_model(name, value):
        if destination:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(value, path)

    representations = {name: arrays[f"h_{name}"] for name in (*heads, "concat")}
    baselines = {name.removeprefix("baseline_"): value for name, value in arrays.items()
                 if name.startswith("baseline_") and name != "baseline_names_json"}
    metrics = {"protocol": {"split_source": "persisted_manifest", "scaler_fit_split": "train",
                           "classifier": "logistic_regression", "classifier_c": classifier_c,
                           "regression_probe": "linear_ridge_on_train_standardized_targets",
                           "eta_probe": "linear_logits_softmax_train_KL_no_target_standardization",
                           "probe_alpha": probe_alpha, "seed": seed,
                           "hyperparameter_selection": "fixed_before_evaluation",
                           "split_counts": {role: int(mask.sum()) for role, mask in masks.items()}},
               "classification": {}, "baselines": {}, "physics_probes": {},
               "network_readouts": {}, "diagnostics": {}}
    predictions = {"event_key": arrays["event_key"], "split": roles, "label": labels}
    fitted = {}
    for group, spaces in (("classification", representations), ("baselines", baselines)):
        for name, features in spaces.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=classifier_c, max_iter=2000, random_state=seed))
            model.fit(features[masks["train"]], labels[masks["train"]])
            probability = model.predict_proba(features)[:, 1]
            predicted = model.predict(features)
            metrics[group][name] = {}
            for role, mask in masks.items():
                metrics[group][name][role] = {
                    "accuracy": float(accuracy_score(labels[mask], predicted[mask])),
                    "balanced_accuracy": float(balanced_accuracy_score(labels[mask], predicted[mask])),
                    "roc_auc": float(roc_auc_score(labels[mask], probability[mask])),
                    "confusion_matrix": confusion_matrix(labels[mask], predicted[mask], labels=[0, 1]).tolist(),
                }
            predictions[f"{group}_{name}_probability"] = probability
            fitted[f"{group}/{name}"] = model
            save_model(f"{group}/{name}.joblib", model)

    for name, features in representations.items():
        metrics["physics_probes"][name] = {}
        for task in TASK_ORDER:
            target = arrays[f"target_{task}"]
            probe = fit_physics_probe(features[masks["train"]], target[masks["train"]], task, alpha=probe_alpha)
            prediction = predict_physics_probe(probe, features)
            metrics["physics_probes"][name][task] = {
                role: physics_metrics(prediction[mask], target[mask], task) for role, mask in masks.items()
            }
            predictions[f"probe_{name}_{task}"] = prediction
            fitted[f"physics_probes/{name}/{task}"] = probe
            save_model(f"physics_probes/{name}/{task}.joblib", probe)

    for task in TASK_ORDER:
        if f"prediction_{task}" not in arrays:
            continue
        saved_prediction = arrays[f"prediction_{task}"]
        if task == "eta":
            prediction = _softmax(saved_prediction)
        else:
            if target_stats is None or task not in target_stats:
                raise ValueError("Saved training target statistics are required to invert network readouts.")
            stats = target_stats[task]
            prediction = saved_prediction * np.asarray(stats["scale"]) + np.asarray(stats["mean"])
        predictions[f"network_{task}"] = prediction
        metrics["network_readouts"][task] = {
            role: physics_metrics(prediction[mask], arrays[f"target_{task}"][mask], task)
            for role, mask in masks.items()
        }

    for role, mask in masks.items():
        metrics["diagnostics"][role] = {
            "h": {name: representation_diagnostics(features[mask]) for name, features in representations.items()},
            "z": {name: representation_diagnostics(arrays[f"z_{name}"][mask]) for name in heads if f"z_{name}" in arrays},
            "cross_head": cross_head_diagnostics({name: representations[name][mask] for name in heads}),
        }
    if destination:
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(destination / "predictions.npz", **predictions)
        # The model's trained readouts are a different experiment from fitting
        # the same probe to every frozen space; keep standalone result files.
        (destination / "network_readouts.json").write_text(
            json.dumps(metrics["network_readouts"], indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(destination / "network_readout_predictions.npz", **{
            key: value for key, value in predictions.items()
            if key in {"event_key", "split"} or key.startswith("network_")
        })
    return {"metrics": metrics, "models": fitted, "predictions": predictions}
