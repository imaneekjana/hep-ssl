#!/usr/bin/env python3
"""Generate the fixed 51-run augmentation study; never connect or submit jobs."""
import argparse
import copy
import csv
import hashlib
from datetime import datetime
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import tarfile
import uuid

try:
    from .runtime_utils import sha256_file, write_json
except ImportError:
    from runtime_utils import sha256_file, write_json

PAIRS = (("ggf", "ttbar"), ("ggf", "dihiggs"), ("ttbar", "dihiggs"))
TRANSFORMS = ("rotate", "energy_noise", "xyz_noise", "shift", "crop")
LETTERS = dict(zip(TRANSFORMS, "rexsc"))
STRENGTHS = dict(zip(TRANSFORMS, ("rotation", "energy_noise", "xyz_noise", "shift_std", "crop_fraction")))
VALUES = dict(zip(TRANSFORMS, (0.3926990817, 0.0001, 5.0, 2.0, 0.5)))
FIXED = {
    "mode": "five_anisotropic_physics", "prepared_dir": None,
    "data": {"events_per_channel": 2500, "pileup": "pu0", "split_seed": 42},
    "grid": {"n_eta": 32, "n_phi": 32},
    "model": {"hidden_dim": 16, "latent_dim": 64, "proj_dim": 32, "k": 8,
              "space_dim": 4, "propagate_dim": 16},
    "objective": {"tau": 0.07, "gamma": 1.0},
    "training": {"epochs": 18, "batch_size": 32, "lr": 0.0003, "weight_decay": 0.0001,
                 "seed": 42, "augmentation_seed": 142, "validation_seed": 242,
                 "device": "cuda", "amp": False},
}


def matrix_configs(base):
    template = copy.deepcopy(base)
    for key, value in FIXED.items():
        if isinstance(value, dict):
            template[key].update(value)
        else:
            template[key] = value
    subsets = [()] + [subset for size in (3, 4, 5) for subset in combinations(TRANSFORMS, size)]
    result = []
    for pair in PAIRS:
        for subset in subsets:
            config = copy.deepcopy(template)
            config["data"]["channels"] = list(pair)
            config["augmentation"].update({"order": list(subset), "rotation_mode": "uniform"})
            for transform, field in STRENGTHS.items():
                config["augmentation"][field] = VALUES[transform] if transform in subset else 0
            suffix = "".join(LETTERS[t] for t in subset) or "none"
            result.append(("_".join(pair) + "_" + suffix, config))
    return result


def differences(before, after, prefix=""):
    result = []
    for key, value in after.items():
        name = prefix + key
        if isinstance(value, dict):
            result.extend(differences(before[key], value, name + "."))
        elif before[key] != value:
            result.append({"field": name, "base": before[key], "adopted": value})
    return result


def read_settings(kit, settings=None):
    value = json.loads((kit / "settings.json").read_text())
    if settings:
        update = settings if isinstance(settings, dict) else json.loads(Path(settings).read_text())
        if set(update) - set(value):
            raise ValueError("Unknown deployment setting")
        value.update(update)
    for key in ("remote_project", "raw_archive", "container_image", "staging_root"):
        path = value[key]
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", path) or ".." in Path(path).parts:
            raise ValueError(f"Unsafe absolute deployment path: {key}")
    if not value["remote_project"].startswith("/home/"):
        raise ValueError("CHTC project/submission directory must be under /home")
    for key in ("raw_archive", "container_image", "staging_root"):
        if not value[key].startswith("/staging/"):
            raise ValueError(f"{key} must use /staging")
    if value["gpu_job_length"] not in ("short", "medium", "long"):
        raise ValueError("Invalid gpu_job_length")
    if value["transfer_protocol"] not in ("osdf", "file"):
        raise ValueError("transfer_protocol must be osdf or file")
    if type(value["request_cpus"]) is not int or value["request_cpus"] < 1:
        raise ValueError("Invalid request_cpus")
    for key in ("request_memory", "prepare_disk", "train_disk"):
        if not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", value[key]):
            raise ValueError(f"Invalid resource size: {key}")
    return value


def staged_url(path, protocol):
    return ("osdf:///chtc" if protocol == "osdf" else "file://") + path


def submit_templates(settings):
    common = f'''universe = vanilla
container_image = {staged_url(settings['container_image'], settings['transfer_protocol'])}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
requirements = (TARGET.HasCHTCStaging == true)
request_cpus = {settings['request_cpus']}
request_memory = {settings['request_memory']}
environment = "OMP_NUM_THREADS={settings['request_cpus']} OPENBLAS_NUM_THREADS={settings['request_cpus']} MKL_NUM_THREADS={settings['request_cpus']} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1"
# manage.py supplies a checked attempt manifest; missing defaults block direct submission.
'''
    prepare = common + f'''manifest = ready_prepare.tsv
attempt = attempts/unconfigured
executable = run_prepare.sh
arguments = $(pair_id) $(config_basename) {Path(settings['raw_archive']).name} $(prepared_name) $(receipt_basename)
transfer_input_files = hep_ssl-code.tar.gz, $(config_path), prepare_data.py, runtime_utils.py, input_paths.json, {staged_url(settings['raw_archive'], settings['transfer_protocol'])}
transfer_output_files = $(prepared_name), prepare_$(pair_id)_status.json, $(receipt_basename)
transfer_output_remaps = "$(prepared_name) = $(prepared_url); prepare_$(pair_id)_status.json = $(attempt)/prepare_$(pair_id)_status.json; $(receipt_basename) = $(attempt)/$(receipt_basename)"
request_disk = {settings['prepare_disk']}
log = $(attempt)/prepare_$(pair_id)_$(Cluster)_$(Process).log
output = $(attempt)/prepare_$(pair_id)_$(Cluster)_$(Process).out
error = $(attempt)/prepare_$(pair_id)_$(Cluster)_$(Process).err
queue pair_id,config_path,config_basename,prepared_name,prepared_url,receipt_basename from $(manifest)
'''
    train = common + f'''manifest = ready_train.tsv
attempt = attempts/unconfigured
stop_after_epoch = 0
resume_basename = -
resume_transfer =
executable = run_gpu.sh
arguments = $(run_id) $(config_basename) $(prepared_name) $(receipt_basename) $(stop_after_epoch) $(resume_basename)
transfer_input_files = hep_ssl-code.tar.gz, $(config_path), gpu_worker.py, runtime_utils.py, $(receipt_path), $(prepared_url) $(resume_transfer)
transfer_output_files = result_$(run_id).tar.gz, status_$(run_id).json
transfer_output_remaps = "result_$(run_id).tar.gz = $(attempt)/result_$(run_id).tar.gz; status_$(run_id).json = $(attempt)/status_$(run_id).json"
request_gpus = 1
+WantGPULab = true
+GPUJobLength = "{settings['gpu_job_length']}"
request_disk = {settings['train_disk']}
log = $(attempt)/train_$(run_id)_$(Cluster)_$(Process).log
output = $(attempt)/train_$(run_id)_$(Cluster)_$(Process).out
error = $(attempt)/train_$(run_id)_$(Cluster)_$(Process).err
queue run_id,config_path,config_basename,prepared_name,prepared_url,receipt_path,receipt_basename from $(manifest)
'''
    return prepare, train


def verification_template(settings):
    prepare, _ = submit_templates(settings)
    common = prepare[:prepare.index('manifest =')]
    return common + f'''manifest = ready_verify.tsv
attempt = attempts/unconfigured
executable = run_prepare.sh
arguments = $(pair_id) $(config_basename) $(prepared_name) $(prepared_name) $(receipt_basename) verify
transfer_input_files = hep_ssl-code.tar.gz, $(config_path), prepare_data.py, runtime_utils.py, input_paths.json, $(prepared_url)
transfer_output_files = prepare_$(pair_id)_status.json, $(receipt_basename)
transfer_output_remaps = "prepare_$(pair_id)_status.json = $(attempt)/prepare_$(pair_id)_status.json; $(receipt_basename) = $(attempt)/$(receipt_basename)"
request_disk = {settings['prepare_disk']}
log = $(attempt)/verify_$(pair_id)_$(Cluster)_$(Process).log
output = $(attempt)/verify_$(pair_id)_$(Cluster)_$(Process).out
error = $(attempt)/verify_$(pair_id)_$(Cluster)_$(Process).err
queue pair_id,config_path,config_basename,prepared_name,prepared_url,receipt_basename from $(manifest)
'''


def build(project, kit=None, output=None, settings=None, reuse_deployment=None):
    project = Path(project).expanduser().resolve()
    kit = Path(kit or Path(__file__).parent).resolve()
    spec = importlib.util.spec_from_file_location("sweep_project_config", project / "src/config.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    base = parser.load_config(project / "configs/pairwise_base.json")
    configs = matrix_configs(base)
    for _, config in configs:
        parser.validate_config(config)
    settings = read_settings(kit, settings)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_augmentation_" + uuid.uuid4().hex[:6]
    destination = Path(output).resolve() if output else project / "deployment" / tag
    if not destination.is_relative_to(project / "deployment") or destination == project / "deployment":
        raise ValueError("Generated output must be a new directory under project/deployment/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", destination.name):
        raise ValueError("Deployment name may not contain whitespace or shell delimiters")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "configs").mkdir()
    (destination / "attempts").mkdir()
    manifest, pairs, overrides = [], {}, {}
    for run_id, config in configs:
        config_path = f"configs/{run_id}.json"
        write_json(destination / config_path, config)
        pair_id = "_".join(config["data"]["channels"])
        row = {"run_id": run_id, "channel_a": config["data"]["channels"][0],
               "channel_b": config["data"]["channels"][1],
               "augmentation_order": config["augmentation"]["order"],
               "config_path": config_path, "config_basename": Path(config_path).name,
               "config_sha256": sha256_file(destination / config_path),
               "prepared_id": pair_id, "seed": config["training"]["seed"],
               "result_file": f"result_{run_id}.tar.gz", "status_file": f"status_{run_id}.json",
               "log_prefix": f"train_{run_id}"}
        manifest.append(row)
        overrides[run_id] = differences(base, config)
        pairs.setdefault(pair_id, {"config_path": config_path, "config_sha256": row["config_sha256"]})
    write_json(destination / "manifest.json", manifest)
    with (destination / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        for row in manifest:
            writer.writerow({**row, "augmentation_order": "+".join(row["augmentation_order"]) or "none"})
    write_json(destination / "overrides.json", overrides)
    write_json(destination / "input_paths.json", {})
    write_json(destination / "prepared_registry.json", {})
    for name in ("run_prepare.sh", "prepare_data.py", "run_gpu.sh", "gpu_worker.py",
                 "runtime_utils.py", "manage.py", "deployment_checks.py", "README_ZH.md",
                 "TESTING.md", "CHANGES_ZH.md"):
        shutil.copy2(kit / name, destination / name)
    for path in destination.glob("*.sh"):
        path.chmod(0o755)
    def archive_filter(member):
        if any(p in {"__pycache__", ".ipynb_checkpoints", ".DS_Store"} for p in Path(member.name).parts):
            return None
        if member.name.startswith("src/figures") or member.name.endswith((".pyc", ".pyo")):
            return None
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Source archive contains unsupported link/special file: {member.name}")
        return member
    with tarfile.open(destination / "hep_ssl-code.tar.gz", "w:gz") as archive:
        for folder in ("src", "configs", "tests", "docs", "chtc_phase1_steps"):
            if (project / folder).exists():
                archive.add(project / folder, arcname=folder, filter=archive_filter)
    source = destination / "hep_ssl-code.tar.gz"
    if source.stat().st_size >= 1_000_000_000:
        raise ValueError("Code archive exceeds 1 GB; exclude non-code artifacts")
    info = {"schema_version": 1, "tag": destination.name, "settings": settings, "pairs": pairs,
            "source_sha256": sha256_file(source), "manifest_sha256": sha256_file(destination / "manifest.json"),
            "stage_directory": settings["staging_root"] + "/" + destination.name,
            "remote_directory": settings["remote_project"] + "/deployment/" + destination.name,
            "run_count": len(manifest)}
    preparation_digest = hashlib.sha256()
    for name in ("src/config.py", "src/prepare_pairwise.py", "src/data/events.py",
                 "src/data/projection.py", "src/physics/targets.py", "chtc_phase1_steps/prepare_data.py"):
        preparation_digest.update(name.encode())
        preparation_digest.update((project / name).read_bytes())
    info['preparation_source_sha256'] = preparation_digest.hexdigest()
    write_json(destination / "deployment.json", info)
    (destination / "source_sha256.txt").write_text(info["source_sha256"] + "  hep_ssl-code.tar.gz\n")
    prepare, train = submit_templates(settings)
    (destination / "01_prepare.sub").write_text(prepare)
    (destination / "02_train.sub").write_text(train)
    (destination / "01_verify.sub").write_text(verification_template(settings))
    if reuse_deployment:
        try:
            from .deployment_checks import reuse_prepared
        except ImportError:
            from deployment_checks import reuse_prepared
        reuse_prepared(destination, Path(reuse_deployment).resolve())
    return destination


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--kit")
    ap.add_argument("--output")
    ap.add_argument("--settings", help="JSON overrides for centralized CHTC paths/resources")
    ap.add_argument("--reuse-deployment", help="Reuse verified prepared outputs from a previous sweep")
    args = ap.parse_args()
    print(build(args.project, args.kit, args.output, args.settings, args.reuse_deployment))
