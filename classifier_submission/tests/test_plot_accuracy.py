"""Small synthetic JSON fixtures for classifier-only accuracy postprocessing.

These tests never inspect a real batch, submit jobs, load models or train.
Only one test writes the complete nine PNG/PDF figure pairs.
"""

import csv
import importlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile

import pytest


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT / "classifier_submission" / "plot_accuracy.py"
PAIRS = {
    "ggf_ttbar": "Single-Higgs (gluon fusion) vs. Top-quark pair",
    "ggf_dihiggs": "Single-Higgs (gluon fusion) vs. Higgs pair",
    "ttbar_dihiggs": "Top-quark pair vs. Higgs pair",
}
REPRESENTATIONS = {
    "general": ("General event features", 64),
    "energy": ("Energy scale", 64),
    "eta": ("Pseudorapidity energy profile", 64),
    "phi": ("Azimuthal structure", 64),
    "local": ("Multiscale energy correlations", 64),
    "concat": ("Combined representation", 320),
}
TRANSFORMS = {
    "r": ("rotate", "Rotation"),
    "e": ("energy_noise", "Energy noise"),
    "x": ("xyz_noise", "Hit-position jitter"),
    "s": ("shift", "Global transverse shift"),
    "c": ("crop", "Spatial masking"),
}
AUGMENTATIONS = ("none", "rex", "res", "rec", "rxs", "rxc", "rsc", "exs", "exc", "esc", "xsc",
                 "rexs", "rexc", "resc", "rxsc", "exsc", "rexsc")
SPLITS = ("train", "val", "test")
LONG_FIELDS = {"run_id", "pair_id", "task_label", "augmentation", "augmentation_label", "n_augmentations",
               "representation", "representation_label", "embedding_dim", "split", "accuracy",
               "accuracy_percent", "none_accuracy", "delta_accuracy_pp", "prepared_fingerprint", "source_file"}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def normalized(text):
    return " ".join(text.split())


def augmentation_label(suffix):
    return "No augmentation" if suffix == "none" else " + ".join(TRANSFORMS[c][1] for c in suffix)


@pytest.fixture
def plotter(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    return importlib.import_module("classifier_submission.plot_accuracy")


@pytest.fixture
def synthetic_batch(tmp_path):
    root = tmp_path / "synthetic classifier batch"
    results = root / "results"
    results.mkdir(parents=True)
    runs = []
    for pair_index, pair in enumerate(PAIRS):
        for aug_index, suffix in enumerate(AUGMENTATIONS):
            run_id = f"{pair}_{suffix}"
            runs.append({"run_id": run_id, "pair_id": pair,
                         "prepared_fingerprint": f"synthetic-fixture-{pair}",
                         # A pretraining path is intentionally unusable: this is
                         # not an allowed source of classifier accuracies.
                         "result_path": "/not-a-classifier-result/DO-NOT-READ.tar.gz"})
            effect = 0 if aug_index == 0 else (-0.024 if aug_index % 2 else 0.013)
            classification = {
                representation: {
                    split: {"accuracy": 0.78 + 0.01 * pair_index + 0.003 * rep_index
                            + 0.001 * split_index + effect + 0.0000000001234567}
                    for split_index, split in enumerate(SPLITS)
                }
                for rep_index, representation in enumerate(REPRESENTATIONS)
            }
            dump(results / f"classifier_status_{run_id}.json", {
                "run_id": run_id, "success": True, "exit_code": 0, "archive_exit_code": 0,
                "classification": classification,
                "physics_probes": {"not_a_representation": 0.999},
                "baselines": {"not_a_representation": 0.999},
            })
    dump(root / "evaluation_plan.json", {"test_fixture": True, "runs": runs})
    return root


def status_path(root, run_id="ggf_ttbar_rex"):
    return root / "results" / f"classifier_status_{run_id}.json"


def find_row(rows, run_id="ggf_ttbar_rex", representation="concat", split="test"):
    matches = [row for row in rows if (row["run_id"], row["representation"], row["split"]) == (run_id, representation, split)]
    assert len(matches) == 1
    return matches[0]


def add_archive(root, run_id, classification):
    path = root / "results" / f"classifier_{run_id}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        members = {
            f"{run_id}/metrics.json": json.dumps({"classification": classification}).encode(),
            f"{run_id}/checkpoints/model.pt": b"Not a model: irrelevant synthetic member.",
            f"{run_id}/classifier.joblib": b"Not a classifier: irrelevant synthetic member.",
            "../must-not-extract.txt": b"Irrelevant unsafe path: must never be extracted.",
        }
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_full_json_only_batch_preserves_precision_ids_labels_and_negative_percentage_points(plotter, synthetic_batch):
    before = {path: path.read_bytes() for path in synthetic_batch.rglob("*.json")}
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert report["counts"] == {"expected": 51, "success": 51, "failed": 0, "missing": 0}
    assert report["valid_accuracy_counts"] == dict.fromkeys(SPLITS, 306)
    assert len(rows) == 918
    assert set(rows[0]) >= LONG_FIELDS
    assert [(pair, suffix) for pair, suffix in dict.fromkeys((r["pair_id"], r["augmentation"]) for r in rows)] == [
        (pair, suffix) for pair in PAIRS for suffix in AUGMENTATIONS
    ]
    for row in rows:
        assert row["task_label"] == PAIRS[row["pair_id"]]
        assert row["augmentation_label"] == augmentation_label(row["augmentation"])
        label, dimensions = REPRESENTATIONS[row["representation"]]
        assert row["representation_label"] == label
        assert row["embedding_dim"] == dimensions
        assert row["n_augmentations"] == (0 if row["augmentation"] == "none" else len(row["augmentation"]))
        assert row["prepared_fingerprint"] == f"synthetic-fixture-{row['pair_id']}"
        source = json.loads(status_path(synthetic_batch, row["run_id"]).read_text())
        assert row["accuracy"] == source["classification"][row["representation"]][row["split"]]["accuracy"]
        assert row["accuracy_percent"] == row["accuracy"] * 100
    row = find_row(rows)
    none = find_row(rows, "ggf_ttbar_none")
    assert row["none_accuracy"] == none["accuracy"]
    assert row["delta_accuracy_pp"] == pytest.approx(-2.4)
    assert row["delta_accuracy_pp"] == 100 * (row["accuracy"] - none["accuracy"])
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(synthetic_batch.rglob("*.tar.gz"))


def test_public_display_mappings_are_complete_and_in_fixed_order(plotter):
    assert dict(plotter.PAIRS) == PAIRS
    assert list(plotter.REPRESENTATIONS) == list(REPRESENTATIONS)
    for key, (label, dimensions) in REPRESENTATIONS.items():
        assert plotter.REPRESENTATIONS[key] == {"label": label, "dim": dimensions}
    assert list(plotter.AUGMENTATIONS) == list(AUGMENTATIONS)
    for suffix in AUGMENTATIONS:
        expected = () if suffix == "none" else tuple(TRANSFORMS[c][0] for c in suffix)
        assert tuple(plotter.AUGMENTATIONS[suffix]) == expected
    assert tuple(plotter.SPLITS) == SPLITS


@pytest.mark.parametrize("problem", ["missing", "failed", "archive_failed", "wrong_id", "broken_json"])
def test_missing_or_failed_status_remains_na_without_using_old_archive(plotter, synthetic_batch, monkeypatch, problem):
    path = status_path(synthetic_batch)
    original = json.loads(path.read_text())
    add_archive(synthetic_batch, "ggf_ttbar_rex", original["classification"])
    if problem == "missing":
        path.unlink()
    elif problem == "broken_json":
        path.write_text("{not valid JSON")
    else:
        original.update({"failed": {"success": False, "exit_code": 1},
                         "archive_failed": {"archive_exit_code": 1},
                         "wrong_id": {"run_id": "ggf_dihiggs_rex"}}[problem])
        dump(path, original)
    monkeypatch.setattr(tarfile, "open", lambda *args, **kwargs: pytest.fail("Failed/missing status must not fall back to an archive."))
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert len(rows) == 918
    invalid = [row for row in rows if row["run_id"] == "ggf_ttbar_rex"]
    assert len(invalid) == 18
    assert all(row["accuracy"] is None and row["accuracy_percent"] is None and row["delta_accuracy_pp"] is None for row in invalid)
    key = "missing" if problem == "missing" else "failed"
    assert report["counts"][key] == 1
    assert report["counts"]["success"] == 50
    assert report["valid_accuracy_counts"] == dict.fromkeys(SPLITS, 300)
    entry = next(run for run in report["runs"] if run["run_id"] == "ggf_ttbar_rex")
    assert entry["status"] == key
    assert entry["reasons"]


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), -0.01, 1.01, "0.8", True])
def test_invalid_single_cell_is_na_while_other_cells_survive(plotter, synthetic_batch, bad_value):
    path = status_path(synthetic_batch)
    status = json.loads(path.read_text())
    status["classification"]["energy"]["val"]["accuracy"] = bad_value
    dump(path, status)
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert find_row(rows, representation="energy", split="val")["accuracy"] is None
    assert find_row(rows)["accuracy"] is not None
    assert report["counts"]["failed"] == 1
    assert report["valid_accuracy_counts"] == {"train": 306, "val": 305, "test": 306}


def test_missing_none_does_not_borrow_baseline_from_another_pair_or_space(plotter, synthetic_batch):
    path = status_path(synthetic_batch, "ggf_ttbar_none")
    status = json.loads(path.read_text())
    del status["classification"]["energy"]["test"]["accuracy"]
    dump(path, status)
    rows, _ = plotter.collect_accuracy(synthetic_batch)
    for row in rows:
        if row["pair_id"] == "ggf_ttbar" and row["representation"] == "energy" and row["split"] == "test":
            assert row["none_accuracy"] is None and row["delta_accuracy_pp"] is None
        elif row["augmentation"] != "none":
            assert row["none_accuracy"] is not None
    assert find_row(rows, representation="energy")["accuracy"] is not None
    path.unlink()
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert report["counts"]["missing"] == 1
    assert all(row["delta_accuracy_pp"] is None for row in rows if row["pair_id"] == "ggf_ttbar")


def test_success_without_classification_reads_only_matching_metrics_json(plotter, synthetic_batch, monkeypatch):
    run_id = "ggf_ttbar_rex"
    path = status_path(synthetic_batch, run_id)
    status = json.loads(path.read_text())
    classification = status.pop("classification")
    dump(path, status)
    archive = add_archive(synthetic_batch, run_id, classification)
    before = {p: p.read_bytes() for p in synthetic_batch.rglob("*") if p.is_file()}
    calls = []
    original = tarfile.TarFile.extractfile

    def read_json_only(self, member):
        name = member.name if isinstance(member, tarfile.TarInfo) else member
        assert name == f"{run_id}/metrics.json"
        calls.append(name)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", read_json_only)
    monkeypatch.setattr(tarfile.TarFile, "extract", lambda *args, **kwargs: pytest.fail("Archive extraction is forbidden."))
    monkeypatch.setattr(tarfile.TarFile, "extractall", lambda *args, **kwargs: pytest.fail("Archive extraction is forbidden."))
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert calls == [f"{run_id}/metrics.json"]
    assert report["counts"]["success"] == 51
    row = find_row(rows)
    assert row["accuracy"] == classification["concat"]["test"]["accuracy"]
    assert archive.name in row["source_file"]
    assert f"{run_id}/metrics.json" in row["source_file"]
    after = {p: p.read_bytes() for p in synthetic_batch.rglob("*") if p.is_file()}
    assert before == after


def test_present_but_incomplete_classification_never_uses_archive(plotter, synthetic_batch, monkeypatch):
    path = status_path(synthetic_batch)
    status = json.loads(path.read_text())
    add_archive(synthetic_batch, "ggf_ttbar_rex", status["classification"])
    status["classification"] = {}
    dump(path, status)
    monkeypatch.setattr(tarfile, "open", lambda *args, **kwargs: pytest.fail("Only an absent classification key allows fallback."))
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert report["counts"]["failed"] == 1
    assert all(row["accuracy"] is None for row in rows if row["run_id"] == "ggf_ttbar_rex")


def test_conflicting_prepared_fingerprints_mark_entire_pair_failed(plotter, synthetic_batch):
    path = synthetic_batch / "evaluation_plan.json"
    plan = json.loads(path.read_text())
    plan["runs"][1]["prepared_fingerprint"] = "different-prepared-fixture"
    dump(path, plan)
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert report["counts"] == {"expected": 51, "success": 34, "failed": 17, "missing": 0}
    assert all(row["accuracy"] is None for row in rows if row["pair_id"] == "ggf_ttbar")
    assert report["valid_accuracy_counts"] == dict.fromkeys(SPLITS, 204)
    assert any("fingerprint" in " ".join(run["reasons"]).lower() for run in report["runs"])


@pytest.mark.parametrize("problem", ["missing_run", "duplicate_run", "wrong_pair", "unknown_run"])
def test_invalid_expected_plan_fails_clearly(plotter, synthetic_batch, problem):
    path = synthetic_batch / "evaluation_plan.json"
    plan = json.loads(path.read_text())
    if problem == "missing_run":
        plan["runs"].pop()
    elif problem == "duplicate_run":
        plan["runs"][-1] = plan["runs"][0].copy()
    elif problem == "wrong_pair":
        plan["runs"][0]["pair_id"] = "ggf_dihiggs"
    else:
        plan["runs"][0]["run_id"] = "ggf_ttbar_unknown"
    dump(path, plan)
    with pytest.raises(ValueError) as error:
        plotter.collect_accuracy(synthetic_batch)
    assert str(error.value)


def test_single_split_batch_has_306_rows_without_inventing_other_splits(plotter, synthetic_batch):
    for path in (synthetic_batch / "results").glob("*.json"):
        status = json.loads(path.read_text())
        for representation in status["classification"]:
            scores = status["classification"][representation]
            status["classification"][representation] = {"test": scores["test"]}
        dump(path, status)
    rows, report = plotter.collect_accuracy(synthetic_batch)
    assert len(rows) == 306
    assert {row["split"] for row in rows} == {"test"}
    assert report["available_splits"] == ["test"]
    assert report["valid_accuracy_counts"] == {"train": 0, "val": 0, "test": 306}
    assert report["counts"]["success"] == 51


def test_figure_names_labels_dimensions_scales_and_longest_label_bounds(plotter, synthetic_batch):
    rows, _ = plotter.collect_accuracy(synthetic_batch)
    figures = plotter.create_figures(rows, "test")
    import matplotlib.pyplot as plt

    try:
        assert len(figures) == 9
        assert {stem for stem, _ in figures} == {f"{kind}_{pair}_test" for pair in PAIRS for kind in ("concat_accuracy", "space_accuracy", "concat_delta")}
        delta_limits = []
        for stem, fig in figures:
            axis = next(ax for ax in fig.axes if len(ax.get_yticklabels()) == 17)
            labels = [normalized(label.get_text()) for label in axis.get_yticklabels()]
            assert labels == [augmentation_label(suffix) for suffix in AUGMENTATIONS]
            if stem.startswith("space_accuracy"):
                columns = [normalized(label.get_text()) for label in axis.get_xticklabels()]
                assert len(columns) == 6
                for column, (label, dimensions) in zip(columns, REPRESENTATIONS.values()):
                    assert label in column and f"{dimensions}D" in column
                mappable = list(axis.images) or list(axis.collections)
                assert mappable[0].get_clim() == (0, 100)
            elif stem.startswith("concat_accuracy"):
                assert axis.get_xlim() == (0, 100)
            else:
                limits = axis.get_xlim()
                assert limits[0] == pytest.approx(-limits[1])
                delta_limits.append(limits)
        assert all(limits == delta_limits[0] for limits in delta_limits)
        # Draw one full heatmap once and verify every tick/title fits the actual
        # canvas, including the five-transform label and long task heading.
        fig = dict(figures)["space_accuracy_ggf_ttbar_test"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        from matplotlib.text import Text
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text().strip():
                continue
            bbox = text.get_window_extent(renderer)
            assert bbox.x0 >= -1 and bbox.y0 >= -1 and bbox.x1 <= width + 1 and bbox.y1 <= height + 1, text.get_text()
    finally:
        for _, fig in figures:
            plt.close(fig)


def png_dpi(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        while True:
            length = struct.unpack(">I", stream.read(4))[0]
            kind = stream.read(4)
            data = stream.read(length)
            stream.read(4)
            if kind == b"pHYs":
                x, y, units = struct.unpack(">IIB", data)
                assert units == 1
                return x * 0.0254, y * 0.0254
            assert kind != b"IEND", "PNG is missing its physical-resolution metadata."


def test_cli_json_only_batch_writes_nine_png_pdf_pairs_and_tables_once(synthetic_batch, tmp_path):
    before = {path: path.read_bytes() for path in synthetic_batch.rglob("*.json")}
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "cli-matplotlib-cache")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--classifier-dir", str(synthetic_batch), "--split", "test"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    output = synthetic_batch / "visualization"
    expected = {f"{kind}_{pair}_test" for pair in PAIRS for kind in ("concat_accuracy", "space_accuracy", "concat_delta")}
    assert {path.stem for path in output.glob("*.png")} == expected
    assert {path.stem for path in output.glob("*.pdf")} == expected
    for path in output.glob("*.png"):
        assert png_dpi(path) == pytest.approx((300, 300), abs=0.01)
    for path in output.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF-")
    with (output / "accuracy_long.csv").open(newline="") as stream:
        long = list(csv.DictReader(stream))
    with (output / "accuracy_test_wide.csv").open(newline="") as stream:
        wide = list(csv.DictReader(stream))
    assert len(long) == 918
    assert sum(row["split"] == "test" for row in long) == 306
    assert len(wide) == 51
    assert set(long[0]) >= LONG_FIELDS
    for representation in REPRESENTATIONS:
        assert any(name.startswith(representation + "_") and "percent" in name for name in wide[0])
    row = find_row(long)
    source = json.loads(status_path(synthetic_batch).read_text())["classification"]["concat"]["test"]["accuracy"]
    assert float(row["accuracy"]) == source
    assert float(row["delta_accuracy_pp"]) == pytest.approx(-2.4)
    report = json.loads((output / "read_report.json").read_text())
    assert report["counts"] == {"expected": 51, "success": 51, "failed": 0, "missing": 0}
    assert report["valid_accuracy_counts"] == dict.fromkeys(SPLITS, 306)
    assert str(synthetic_batch) in result.stdout and str(output) in result.stdout
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(synthetic_batch.rglob("*.tar.gz"))
