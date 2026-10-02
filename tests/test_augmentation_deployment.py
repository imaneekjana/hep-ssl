"""The reviewed 3-pair × 17-subset sweep, without cluster submission.

Expected combinations are independently enumerated and historical parameter
values come from the user-supplied CSV fixture, not the deployment generator.
"""

from collections import Counter
import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from chtc_phase1_steps import build_deployment as deployment
from chtc_phase1_steps import deployment_checks as checks
from chtc_phase1_steps import manage
from chtc_phase1_steps.runtime_utils import metadata_fingerprint, sha256_file, write_json
from src.config import load_config, validate_config


PROJECT = Path(__file__).resolve().parents[1]
PAIRS = (("ggf", "ttbar"), ("ggf", "dihiggs"), ("ttbar", "dihiggs"))
ORDER = ("rotate", "energy_noise", "xyz_noise", "shift", "crop")
LETTERS = dict(zip(ORDER, "rexsc"))
PARAMETERS = {
    "rotate": ("rotation", 0.3926990817),
    "energy_noise": ("energy_noise", 0.0001),
    "xyz_noise": ("xyz_noise", 5.0),
    "shift": ("shift_std", 2.0),
    "crop": ("crop_fraction", 0.5),
}


def expected_subsets():
    return [()] + [subset for count in (3, 4, 5) for subset in itertools.combinations(ORDER, count)]


def expected_run_id(pair, subset):
    suffix = "".join(LETTERS[name] for name in subset) if subset else "none"
    return "_".join((*pair, suffix))


@pytest.fixture
def base_config():
    return load_config(PROJECT / "configs" / "pairwise_base.json")


@pytest.fixture
def matrix(base_config):
    return deployment.matrix_configs(base_config)


@pytest.fixture
def built(tmp_path):
    # A small actual project snapshot keeps tests portable and avoids packaging
    # histories, generated runs or local data. Production builder runs unchanged.
    project = tmp_path / "project with spaces"
    project.mkdir()
    for name in ("src", "configs", "docs", "chtc_phase1_steps"):
        shutil.copytree(PROJECT / name, project / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (project / "tests").mkdir()
    shutil.copy2(__file__, project / "tests" / Path(__file__).name)
    shutil.copytree(PROJECT / "tests" / "fixtures", project / "tests" / "fixtures")
    shutil.copy2(PROJECT / ".gitignore", project / ".gitignore")
    output = project / "deployment" / "2026_10_01_test_sweep"
    result = deployment.build(project, output=output)
    assert Path(result) == output
    return project, output


def test_exact_51_runs_in_requested_pair_and_subset_order(matrix):
    ids = [run_id for run_id, _ in matrix]
    expected = [expected_run_id(pair, subset) for pair in PAIRS for subset in expected_subsets()]
    assert ids == expected
    assert len(ids) == len(set(ids)) == 51
    assert all("codex" not in name.lower() and "chatgpt" not in name.lower() for name in ids)


@pytest.mark.parametrize("pair", PAIRS)
def test_each_pair_has_all_subsets_once_in_canonical_order(matrix, pair):
    configs = [config for _, config in matrix if config["data"]["channels"] == list(pair)]
    actual = [tuple(config["augmentation"]["order"]) for config in configs]
    assert len(configs) == 17
    assert Counter(map(len, actual)) == {0: 1, 3: 10, 4: 5, 5: 1}
    assert set(actual) == set(expected_subsets())
    for subset in actual:
        assert tuple(name for name in ORDER if name in subset) == subset


def test_enabled_intensities_and_disabled_zero_values(matrix):
    for run_id, config in matrix:
        aug = config["augmentation"]
        assert aug["rotation_mode"] == "uniform"
        for transform, (parameter, enabled) in PARAMETERS.items():
            expected = enabled if transform in aug["order"] else 0.0
            assert aug[parameter] == pytest.approx(expected, abs=1e-15), (run_id, parameter)
        if run_id.endswith("_none"):
            assert aug["order"] == []
            assert all(aug[field] == 0 for field, _ in PARAMETERS.values())
            assert config["mode"] == "five_anisotropic_physics"
            assert config["objective"] == {"tau": 0.07, "gamma": 1.0}
            assert config["training"]["epochs"] == 18


def test_every_generated_config_passes_existing_validation_with_fixed_design(matrix):
    for _, config in matrix:
        validate_config(copy.deepcopy(config))
        assert config["mode"] == "five_anisotropic_physics"
        assert config["data"]["events_per_channel"] == 2500
        assert config["data"]["pileup"] == "pu0"
        assert config["data"]["split_seed"] == 42
        assert config["grid"]["n_eta"] == config["grid"]["n_phi"] == 32
        assert config["model"] == {"hidden_dim": 16, "latent_dim": 64, "proj_dim": 32,
                                   "k": 8, "space_dim": 4, "propagate_dim": 16}
        assert config["objective"] == {"tau": 0.07, "gamma": 1.0}
        expected = {"epochs": 18, "batch_size": 32, "lr": 0.0003,
                    "weight_decay": 0.0001, "seed": 42, "augmentation_seed": 142,
                    "validation_seed": 242, "device": "cuda", "amp": False}
        assert {key: config["training"][key] for key in expected} == expected


def test_within_pair_only_augmentation_changes_and_generator_does_not_mutate_base(base_config):
    before = copy.deepcopy(base_config)
    matrix = deployment.matrix_configs(base_config)
    assert base_config == before
    for pair in PAIRS:
        configs = [copy.deepcopy(config) for _, config in matrix if config["data"]["channels"] == list(pair)]
        for config in configs:
            config.pop("augmentation")
        assert all(config == configs[0] for config in configs[1:])
    # Editing a returned run must not alias another run or the caller's baseline.
    matrix[0][1]["model"]["hidden_dim"] = 999
    assert matrix[1][1]["model"]["hidden_dim"] == 16
    assert base_config == before


def test_reference_csv_parameters_match_all_51_generated_configs(matrix):
    fieldnames = ["run_id", "channel_a", "channel_b", "events_per_channel", "rotation_mode",
                  "rotation", "energy_noise", "xyz_noise", "shift_std", "crop_fraction",
                  "order", "lr", "weight_decay", "batch_size", "tau", "seed"]
    path = PROJECT / "tests" / "fixtures" / "experiments_reference.txt"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, fieldnames=fieldnames))
    reference = {row["run_id"]: row for row in rows}
    assert len(reference) == len(rows) == 51
    assert set(reference) == {run_id for run_id, _ in matrix}
    for run_id, config in matrix:
        row = reference[run_id]
        assert config["data"]["channels"] == [row["channel_a"], row["channel_b"]]
        assert config["data"]["events_per_channel"] == int(row["events_per_channel"])
        aug = config["augmentation"]
        expected_order = [] if row["order"] == "none" else row["order"].split("+")
        assert aug["order"] == expected_order
        assert aug["rotation_mode"] == row["rotation_mode"]
        for field, _ in PARAMETERS.values():
            assert aug[field] == pytest.approx(float(row[field]), abs=1e-15)
        for field in ("lr", "weight_decay", "batch_size", "seed"):
            assert config["training"][field] == float(row[field])
        assert config["objective"]["tau"] == float(row["tau"])


def test_build_manifest_files_config_hashes_and_shared_pair_mapping(built):
    _, output = built
    manifest = json.loads((output / "manifest.json").read_text())
    info = json.loads((output / "deployment.json").read_text())
    with (output / "manifest.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(manifest) == len(table) == 51
    assert [row["run_id"] for row in table] == [row["run_id"] for row in manifest]
    assert set(info["pairs"]) == {"_".join(pair) for pair in PAIRS}
    assert Counter(row["prepared_id"] for row in manifest) == {"_".join(pair): 17 for pair in PAIRS}
    for row in manifest:
        pair = "_".join((row["channel_a"], row["channel_b"]))
        assert row["prepared_id"] == pair
        assert row["config_path"] == f"configs/{row['run_id']}.json"
        assert row["config_basename"] == f"{row['run_id']}.json"
        path = output / row["config_path"]
        config = load_config(path)
        assert config["data"]["channels"] == [row["channel_a"], row["channel_b"]]
        assert config["augmentation"]["order"] == row["augmentation_order"]
        assert config["training"]["seed"] == row["seed"] == 42
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["config_sha256"]
        assert row["result_file"] == f"result_{row['run_id']}.tar.gz"
        assert row["status_file"] == f"status_{row['run_id']}.json"
        assert row["log_prefix"] == f"train_{row['run_id']}"
    for field in ("run_id", "config_path", "result_file", "status_file", "log_prefix"):
        assert len({row[field] for row in manifest}) == 51
    for pair, detail in info["pairs"].items():
        assert detail["config_path"] == f"configs/{pair}_none.json"
        expected = next(row["config_sha256"] for row in manifest if row["run_id"] == f"{pair}_none")
        assert detail["config_sha256"] == expected
    overrides = json.loads((output / "overrides.json").read_text())
    adopted = {item["field"]: item for item in overrides["ggf_ttbar_res"]}
    assert adopted["augmentation.shift_std"] == {"field": "augmentation.shift_std", "base": 0.0, "adopted": 2.0}
    assert adopted["training.device"]["adopted"] == "cuda"
    assert adopted["augmentation.order"]["adopted"] == ["rotate", "energy_noise", "shift"]
    assert info["settings"]["remote_project"] == "/home/kli398/hep_ssl_chtc"
    assert info["settings"]["raw_archive"] == "/staging/k/kli398/colliderml-data-pairwise-2500.tar.gz"
    assert info["settings"]["container_image"] == "/staging/k/kli398/hep_ssl.sif"
    assert info["remote_directory"] == info["settings"]["remote_project"] + "/deployment/" + output.name


def test_one_archive_has_portable_module_roots_and_no_runtime_outputs(built):
    _, output = built
    archives = list(output.glob("*.tar.gz"))
    assert [path.name for path in archives] == ["hep_ssl-code.tar.gz"]
    info = json.loads((output / "deployment.json").read_text())
    assert hashlib.sha256(archives[0].read_bytes()).hexdigest() == info["source_sha256"]
    with tarfile.open(archives[0], "r:gz") as archive:
        names = archive.getnames()
    assert {Path(name).parts[0] for name in names} == {"src", "configs", "tests", "docs", "chtc_phase1_steps"}
    assert {"src/train_pairwise.py", "src/prepare_pairwise.py", "configs/pairwise_base.json",
            "chtc_phase1_steps/gpu_worker.py", "chtc_phase1_steps/prepare_data.py"} <= set(names)
    for name in names:
        assert not Path(name).is_absolute()
        assert ".." not in Path(name).parts
        assert not {"deployment", "__pycache__", ".pytest_cache", ".git", "outputs", "experiments"}.intersection(Path(name).parts)
        assert not name.endswith((".pyc", ".pt", ".sif", ".tar.gz"))


def test_submit_templates_cannot_launch_before_readiness_manifest_exists(built):
    _, output = built
    assert not (output / "ready_prepare.tsv").exists()
    assert not (output / "ready_train.tsv").exists()
    for filename, default in (("01_prepare.sub", "ready_prepare.tsv"), ("02_train.sub", "ready_train.tsv")):
        content = (output / filename).read_text()
        assert re.search(r"(?m)^\s*manifest\s*=\s*" + re.escape(default) + r"\s*$", content)
        assert re.search(r"(?im)^\s*queue\s+.+\s+from\s+\$\(manifest\)\s*$", content)
        assert not re.search(r"(?im)^\s*queue\s+(?:1|51)\s*$", content)
        assert "/Users/" not in content
    train = (output / "02_train.sub").read_text()
    assert "$(run_id)" in train
    assert "request_gpus = 1" in train
    assert not list(output.glob("*first_epoch*.sub"))
    assert not list(output.glob("*continue*.sub"))


def test_active_shell_wrappers_have_valid_bash_syntax(built):
    _, output = built
    scripts = list((PROJECT / "chtc_phase1_steps").glob("*.sh")) + list(output.glob("*.sh"))
    assert scripts
    for script in scripts:
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_generation_refuses_output_outside_ignored_deployment_tree(built, tmp_path):
    project, _ = built
    for invalid in (tmp_path / "outside", project, project / "deployment"):
        with pytest.raises(ValueError):
            deployment.build(project, output=invalid)


def test_gpu_wrapper_failure_returns_nonzero_and_keeps_error_and_partial_output(tmp_path):
    # A packaging failure is exercised in an isolated execute-node directory.
    # No training or CUDA availability is simulated.
    run_id = "ggf_ttbar_none"
    scratch = tmp_path / "execute node"
    partial = scratch / "outputs" / run_id / "partial.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("Existing partial output must survive a failed attempt.\n")
    shutil.copy2(PROJECT / "chtc_phase1_steps" / "runtime_utils.py", scratch / "runtime_utils.py")
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run([
        "bash", str(PROJECT / "chtc_phase1_steps" / "run_gpu.sh"), run_id,
        f"{run_id}.json", "prepared_ggf_ttbar.tar.gz", "receipt_ggf_ttbar.json", "0", "-",
    ], cwd=scratch, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    status = json.loads((scratch / f"status_{run_id}.json").read_text())
    assert status["success"] is False
    assert status["exit_code"] != 0
    assert status["run_id"] == run_id
    with tarfile.open(scratch / f"result_{run_id}.tar.gz", "r:gz") as archive:
        assert archive.extractfile(f"{run_id}/partial.txt").read().decode() == partial.read_text()
        log = archive.extractfile(f"{run_id}/wrapper.log").read().decode()
    assert "hep_ssl-code.tar.gz" in log


@pytest.fixture
def unit_receipts(built):
    """Mock scheduler receipts only, never a real-data or CUDA validation claim.

    The gate is deliberately lightweight and must consume only returned JSON;
    execute-node tests elsewhere validate actual prepared contents and hashes.
    """
    project, root = built
    info = json.loads((root / "deployment.json").read_text())
    attempt = root / "attempts" / "unit_test_receipts"
    attempt.mkdir()
    registry = {}
    for process, (pair, details) in enumerate(info["pairs"].items()):
        config = json.loads((root / details["config_path"]).read_text())
        manifest = [{"key": f"unit-receipt-only/{pair}", "split": "train"}]
        metadata = {"synthetic": False, "test_fixture": True, "manifest": manifest,
                    "manifest_hash": metadata_fingerprint(manifest),
                    "preparation_config": {name: copy.deepcopy(config[name]) for name in ("data", "grid", "targets")}}
        metadata["data"] = copy.deepcopy(config["data"])
        metadata["data"]["dataset_revision"] = "unit-receipt-revision"
        metadata["preparation_config"]["data"] = copy.deepcopy(metadata["data"])
        metadata["metadata_hash"] = metadata_fingerprint(metadata)
        name = f"prepared_{pair}.tar.gz"
        receipt_path = attempt / f"receipt_{pair}.json"
        status_path = attempt / f"prepare_{pair}_status.json"
        receipt = {"success": True, "synthetic": False, "pair_id": pair, "archive": name,
                   "config_sha256": details["config_sha256"], "code_archive_sha256": info["source_sha256"],
                   "archive_sha256": "a" * 64, "archive_bytes": 100,
                   "preprocessing": metadata, "metadata_fingerprint": metadata_fingerprint(metadata),
                   "metadata_hash": metadata["metadata_hash"], "manifest_hash": metadata["manifest_hash"],
                   "input_paths_sha256": sha256_file(root / "input_paths.json"), "raw_archive_sha256": "c" * 64}
        write_json(receipt_path, receipt)
        status = {key: receipt[key] for key in ("success", "pair_id", "archive", "config_sha256", "code_archive_sha256", "archive_sha256")}
        status.update(exit_code=0, receipt_sha256=sha256_file(receipt_path), archive_bytes=100)
        write_json(status_path, status)
        registry[pair] = {
            "status_path": str(status_path.relative_to(root)), "receipt_path": str(receipt_path.relative_to(root)),
            "prepared_name": name, "prepared_url": f"osdf:///chtc/staging/unit-fixture/{name}",
            "config_sha256": details["config_sha256"], "source_sha256": info["source_sha256"],
            "input_paths_sha256": sha256_file(root / "input_paths.json"),
            "job_id": f"99999.{process}",
            "completion": {"ClusterId": 99999, "ProcId": process, "JobStatus": 4, "ExitCode": 0, "ExitBySignal": False},
        }
    write_json(root / "prepared_registry.json", registry)
    return project, root, registry


def _reseal_unit_receipt(root, record, receipt, *, metadata_changed=False):
    if metadata_changed:
        metadata = receipt["preprocessing"]
        metadata.pop("metadata_hash", None)
        metadata["metadata_hash"] = metadata_fingerprint(metadata)
        receipt["metadata_fingerprint"] = metadata_fingerprint(metadata)
        receipt["metadata_hash"] = metadata["metadata_hash"]
        receipt["manifest_hash"] = metadata["manifest_hash"]
    path = root / record["receipt_path"]
    write_json(path, receipt)
    status_path = root / record["status_path"]
    status = json.loads(status_path.read_text())
    status["receipt_sha256"] = sha256_file(path)
    write_json(status_path, status)


def test_gate_requires_three_preparations_and_successful_scheduler_completion(built, monkeypatch):
    _, root = built
    monkeypatch.setattr(checks, "completed_job", lambda _: pytest.fail("Missing preparation must never query or submit a job."))
    with pytest.raises(RuntimeError, match="no preparation"):
        checks.check_prepared(root, query_history=False)


def test_gate_accepts_verified_unit_receipts_without_contacting_scheduler(unit_receipts, monkeypatch):
    _, root, registry = unit_receipts
    monkeypatch.setattr(checks, "completed_job", lambda _: pytest.fail("Cached completion must not contact HTCondor."))
    result = checks.check_prepared(root, query_history=False)
    assert set(result) == {"_".join(pair) for pair in PAIRS}
    for pair, record in result.items():
        assert record["completion"] == registry[pair]["completion"]
        assert len(record["metadata_fingerprint"]) == 64
        assert record["archive_sha256"] == "a" * 64


@pytest.mark.parametrize("problem,match", [
    ("missing", "have not returned"),
    ("failed", "preparation failed"),
    ("synthetic_receipt", "real prepared data"),
    ("synthetic_metadata", "synthetic data"),
    ("config_digest", "configuration digest"),
    ("archive_digest", "archive digest"),
    ("fingerprint", "fingerprint mismatch"),
    ("split_manifest", "split manifest checksum"),
    ("incompatible_grid", "incompatible prepared grid.n_eta"),
    ("not_completed", "finish successfully"),
    ("uncached_completion", "scheduler completion gate"),
    ("wrong_completion_job", "belongs to another job"),
    ("archive_size", "archive size mismatch"),
    ("empty_archive", "empty archive"),
    ("manifest_receipt", "receipt manifest_hash"),
    ("metadata_receipt", "receipt metadata_hash"),
    ("input_paths", "input-path digest mismatch"),
    ("raw_snapshot", "different raw archive snapshots"),
    ("revision", "different raw archive snapshots"),
])
def test_gate_rejects_missing_failed_synthetic_stale_or_incompatible_outputs(unit_receipts, problem, match):
    _, root, registry = unit_receipts
    pair = "ggf_ttbar"
    record = registry[pair]
    status_path = root / record["status_path"]
    receipt_path = root / record["receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    if problem == "missing":
        receipt_path.unlink()
    elif problem in {"failed", "config_digest", "archive_size"}:
        status = json.loads(status_path.read_text())
        update = {"failed": {"success": False, "exit_code": 1}, "config_digest": {"config_sha256": "b" * 64},
                  "archive_size": {"archive_bytes": 101}}[problem]
        status.update(update)
        write_json(status_path, status)
    elif problem == "not_completed":
        record["completion"]["JobStatus"] = 2
    elif problem == "uncached_completion":
        record.pop("completion")
    elif problem == "wrong_completion_job":
        record["completion"]["ProcId"] = 999
    else:
        if problem == "synthetic_receipt":
            receipt["synthetic"] = True
        elif problem == "synthetic_metadata":
            receipt["preprocessing"]["synthetic"] = True
        elif problem == "archive_digest":
            receipt["archive_sha256"] = "b" * 64
        elif problem == "fingerprint":
            receipt["metadata_fingerprint"] = "b" * 64
        elif problem == "split_manifest":
            receipt["preprocessing"]["manifest"][0]["split"] = "test"
        elif problem == "incompatible_grid":
            receipt["preprocessing"]["preparation_config"]["grid"]["n_eta"] = 16
        elif problem == "empty_archive":
            receipt["archive_bytes"] = 0
        elif problem == "manifest_receipt":
            receipt["manifest_hash"] = "b" * 64
        elif problem == "metadata_receipt":
            receipt["metadata_hash"] = "b" * 64
        elif problem == "input_paths":
            receipt["input_paths_sha256"] = "b" * 64
        elif problem == "raw_snapshot":
            receipt["raw_archive_sha256"] = "b" * 64
        elif problem == "revision":
            receipt["preprocessing"]["data"]["dataset_revision"] = "other-unit-receipt-revision"
            receipt["preprocessing"]["preparation_config"]["data"]["dataset_revision"] = "other-unit-receipt-revision"
        _reseal_unit_receipt(root, record, receipt, metadata_changed=problem in {"synthetic_metadata", "split_manifest", "incompatible_grid", "revision"})
    write_json(root / "prepared_registry.json", registry)
    with pytest.raises((ValueError, RuntimeError), match=match):
        checks.check_prepared(root, query_history=False)


def test_reuse_keeps_same_staged_archive_and_fingerprint(unit_receipts):
    project, previous, registry = unit_receipts
    verified = checks.check_prepared(previous, query_history=False)
    destination = deployment.build(project, output=project / "deployment" / "2026_10_01_reused",
                                   reuse_deployment=previous)
    reused = checks.check_prepared(destination, query_history=False)
    for pair in registry:
        assert reused[pair]["prepared_url"] == verified[pair]["prepared_url"]
        assert reused[pair]["archive_sha256"] == verified[pair]["archive_sha256"]
        assert reused[pair]["metadata_fingerprint"] == verified[pair]["metadata_fingerprint"]
        for key in ("receipt_path", "status_path"):
            assert reused[pair][key].startswith("reused/")
            assert (destination / reused[pair][key]).read_bytes() == (previous / verified[pair][key]).read_bytes()


def test_changed_configuration_cannot_reuse_old_prepared(unit_receipts):
    project, previous, _ = unit_receipts
    path = project / "configs" / "pairwise_base.json"
    base = json.loads(path.read_text())
    base["grid"]["cell_cutoff_gev"] = 0.125
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="incompatible prepared grid.cell_cutoff_gev"):
        deployment.build(project, output=project / "deployment" / "2026_10_01_incompatible",
                         reuse_deployment=previous)


@pytest.mark.parametrize("artifact,match", [("source", "Source archive changed"), ("config", "Config changed"), ("manifest", "Manifest changed")])
def test_gate_detects_changed_deployment_artifacts(built, artifact, match):
    _, root = built
    path = {"source": root / "hep_ssl-code.tar.gz", "config": root / "configs" / "ggf_ttbar_none.json",
            "manifest": root / "manifest.json"}[artifact]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=match):
        checks.verify_deployment(root)


def test_prepare_dry_run_writes_exactly_three_cpu_transfer_rows_without_submission(built, monkeypatch):
    _, root = built
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run must not invoke HTCondor."))
    manage.prepare_jobs(root, dry_run=True)
    attempts = list((root / "attempts").glob("prepare_*"))
    assert len(attempts) == 1
    rows = list(csv.reader((attempts[0] / "prepare.tsv").read_text().splitlines(), delimiter="\t"))
    assert len(rows) == 3
    assert [row[0] for row in rows] == ["_".join(pair) for pair in PAIRS]
    for pair, config_path, basename, archive, url, receipt in rows:
        assert config_path == f"configs/{pair}_none.json"
        assert basename == f"{pair}_none.json"
        assert Path(archive).name == archive
        assert archive != "colliderml-data-pairwise-2500.tar.gz"
        assert url.endswith("/" + archive)
        assert receipt == f"prepare_{pair}_details.json"
    assert json.loads((root / "prepared_registry.json").read_text()) == {}
    assert not (attempts[0] / "submission.json").exists()


def test_training_dry_run_binds_51_distinct_configs_to_three_shared_prepared_inputs(unit_receipts, monkeypatch):
    _, root, registry = unit_receipts
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run must not invoke HTCondor."))
    manage.train_jobs(root, dry_run=True)
    attempts = list((root / "attempts").glob("train_*"))
    assert len(attempts) == 1
    rows = list(csv.reader((attempts[0] / "train.tsv").read_text().splitlines(), delimiter="\t"))
    assert len(rows) == 51
    assert [row[0] for row in rows] == [expected_run_id(pair, subset) for pair in PAIRS for subset in expected_subsets()]
    assert len({row[1] for row in rows}) == len({row[2] for row in rows}) == 51
    assert len({row[3] for row in rows}) == len({row[4] for row in rows}) == len({row[5] for row in rows}) == 3
    for run_id, config_path, basename, archive, url, receipt_path, receipt_basename in rows:
        pair = run_id.rsplit("_", 1)[0]
        assert config_path == f"configs/{run_id}.json"
        assert basename == Path(config_path).name
        assert archive == registry[pair]["prepared_name"]
        assert url == registry[pair]["prepared_url"]
        assert receipt_path == registry[pair]["receipt_path"]
        assert receipt_basename == Path(receipt_path).name
    plan = json.loads((attempts[0] / "plan.json").read_text())
    assert plan["stop_after_epoch"] == 0
    assert len(plan["runs"]) == 51
    assert not (attempts[0] / "submission.json").exists()


def test_optional_short_run_selects_one_run_and_preserves_18_epoch_configuration(unit_receipts, monkeypatch):
    _, root, _ = unit_receipts
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run must not invoke HTCondor."))
    run_id = "ggf_ttbar_none"
    config_path = root / "configs" / f"{run_id}.json"
    config_before = config_path.read_bytes()
    manage.train_jobs(root, run_id=run_id, stop_after_epoch=1, dry_run=True)
    attempt = next((root / "attempts").glob("train_*"))
    rows = list(csv.reader((attempt / "train.tsv").read_text().splitlines(), delimiter="\t"))
    assert len(rows) == 1 and rows[0][0] == run_id
    assert json.loads((attempt / "plan.json").read_text())["stop_after_epoch"] == 1
    assert config_path.read_bytes() == config_before
    assert json.loads(config_before)["training"]["epochs"] == 18
    with pytest.raises(ValueError, match="exactly one"):
        manage.train_jobs(root, stop_after_epoch=1, dry_run=True)


def test_training_gate_precedes_any_submission_attempt(built, monkeypatch):
    _, root = built
    monkeypatch.setattr(manage, "submit", lambda *args, **kwargs: pytest.fail("Unprepared input reached submission."))
    with pytest.raises(RuntimeError, match="no preparation"):
        manage.train_jobs(root, dry_run=True)
    assert not list((root / "attempts").glob("train_*"))


def test_actual_submission_refuses_non_home_deployment_directory(built, monkeypatch):
    _, root = built
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Wrong directory reached condor_submit."))
    with pytest.raises(ValueError, match="Run submission from /home/"):
        manage.prepare_jobs(root, dry_run=False)
    assert not list((root / "attempts").glob("prepare_*"))


def test_repeated_plans_use_separate_attempt_directories(unit_receipts, monkeypatch):
    _, root, _ = unit_receipts
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run must not invoke HTCondor."))
    manage.train_jobs(root, run_id="ggf_ttbar_rex", dry_run=True)
    manage.train_jobs(root, run_id="ggf_ttbar_rex", dry_run=True)
    plans = list((root / "attempts").glob("train_*/plan.json"))
    assert len(plans) == 2
    assert plans[0].parent != plans[1].parent
    assert all((path.parent / "train.tsv").is_file() for path in plans)


def test_existing_prepared_reuse_dry_run_only_schedules_one_cpu_verification(built, monkeypatch):
    _, root = built
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run must not invoke HTCondor."))
    url = "osdf:///chtc/staging/k/kli398/old/prepared_ggf_ttbar.tar.gz"
    manage.reuse_archive(root, pair="ggf_ttbar", archive=url, dry_run=True)
    attempts = list((root / "attempts").glob("verify_*"))
    assert len(attempts) == 1
    rows = list(csv.reader((attempts[0] / "verify.tsv").read_text().splitlines(), delimiter="\t"))
    assert rows == [["ggf_ttbar", "configs/ggf_ttbar_none.json", "ggf_ttbar_none.json",
                     "prepared_ggf_ttbar.tar.gz", url, "prepare_ggf_ttbar_details.json"]]
    plan = json.loads((attempts[0] / "plan.json").read_text())
    assert plan["stage"] == "verify"
    assert plan["pairs"]["ggf_ttbar"]["operation"] == "verify"
    assert plan["pairs"]["ggf_ttbar"]["input_paths_sha256"] is None
    assert json.loads((root / "prepared_registry.json").read_text()) == {}
    assert not (attempts[0] / "submission.json").exists()
    submit = (root / "01_verify.sub").read_text()
    assert "request_gpus" not in submit
    assert "$(prepared_url)" in submit
    assert "queue " in submit and "from $(manifest)" in submit


@pytest.mark.parametrize("url", ["/Users/local/prepared.tar.gz", "https://example.org/prepared.tar.gz",
                                 "osdf:///chtc/staging/k/../raw.tar.gz"])
def test_existing_archive_reuse_rejects_non_staging_or_traversal_url(built, url):
    _, root = built
    with pytest.raises(ValueError):
        manage.reuse_archive(root, pair="ggf_ttbar", archive=url, dry_run=True)
    assert not list((root / "attempts").glob("verify_*"))


def test_changed_preprocessing_source_blocks_automatic_prepared_reuse(unit_receipts):
    project, previous, _ = unit_receipts
    path = project / "src" / "data" / "projection.py"
    path.write_text(path.read_text() + "\n# Unit-test source-change marker.\n")
    with pytest.raises(ValueError, match="Preprocessing source changed"):
        deployment.build(project, output=project / "deployment" / "2026_10_01_changed_source",
                         reuse_deployment=previous)


@pytest.mark.parametrize("ads,allow_removed,accepted", [
    ([], False, False),
    ([{"ClusterId": 99, "ProcId": 0, "JobStatus": 4, "ExitCode": 0}], False, False),
    ([{"ClusterId": 99999, "ProcId": 0, "JobStatus": 2}], False, False),
    ([{"ClusterId": 99999, "ProcId": 0, "JobStatus": 3}], False, False),
    ([{"ClusterId": 99999, "ProcId": 0, "JobStatus": 4, "ExitCode": 0, "ExitBySignal": True}], False, False),
    ([{"ClusterId": 99999, "ProcId": 0, "JobStatus": 3}], True, True),
    ([{"ClusterId": 99999, "ProcId": 0, "JobStatus": 4, "ExitCode": 0}], False, True),
])
def test_scheduler_history_filters_requested_job_and_retry_state(monkeypatch, ads, allow_removed, accepted):
    # Mock read-only scheduler response only; no cluster command is executed.
    calls = []

    def history(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(ads))

    monkeypatch.setattr(checks.subprocess, "run", history)
    if accepted:
        assert checks.completed_job("99999.0", allow_removed=allow_removed) == ads[0]
    else:
        with pytest.raises(RuntimeError):
            checks.completed_job("99999.0", allow_removed=allow_removed)
    assert len(calls) == 1
    assert calls[0][:2] == ["condor_history", "99999.0"]


def test_submit_dynamic_macros_use_append_to_override_template_defaults(built, monkeypatch, capsys):
    _, root = built
    info = json.loads((root / "deployment.json").read_text())
    attempt = manage.attempt_directory(root, "train")
    manifest = attempt / "train.tsv"
    manifest.write_text("unit-test dry-run manifest\n")
    relative = str(attempt.relative_to(root))
    macros = ["stop_after_epoch=1", "resume_basename=resume_input.tar.gz",
              f"resume_transfer=, {relative}/resume_input.tar.gz"]
    monkeypatch.setattr(manage.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run must not invoke HTCondor."))
    assert manage.submit(root, info, attempt, "02_train.sub", manifest, 1,
                         dry_run=True, macros=macros) is None
    command = shlex.split(capsys.readouterr().out.splitlines()[0])
    assert command[:3] == ["condor_submit", "-terse", "02_train.sub"]
    # A bare NAME=value before parsing the template can be overwritten by the
    # file's defaults. Each override must be its own -append argument pair.
    assignments = [f"manifest={relative}/train.tsv", f"attempt={relative}", *macros]
    assert command[3::2] == ["-append"] * len(assignments)
    assert command[4::2] == assignments
