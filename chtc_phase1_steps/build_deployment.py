#!/usr/bin/env python3
"""Create deployment-only files. Uses stdlib; never edits model/data source code."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import uuid


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def build(project, kit, output=None):
    project = Path(project).expanduser().resolve()
    kit = Path(kit).resolve()
    required = ["configs/pairwise_base.json", "src/config.py", "src/data/events.py",
                "src/prepare_pairwise.py", "src/train_pairwise.py",
                "src/training/trainer.py", "src/training/checkpoint.py",
                "src/models/multispace.py", "src/losses/multitask.py"]
    for name in required:
        if not (project / name).is_file():
            raise FileNotFoundError(f"Missing {project / name}; save the new Codex implementation first.")
    config = json.loads((project / "configs/pairwise_base.json").read_text())
    config = copy.deepcopy(config)
    config["training"]["device"] = "cuda"
    config["prepared_dir"] = None
    if config["mode"] != "five_anisotropic_physics":
        raise ValueError("This deployment is for the agreed five_anisotropic_physics mode.")
    if config["training"]["epochs"] < 2:
        raise ValueError("The first-epoch/resume workflow requires epochs >= 2.")
    channels = config["data"]["channels"]
    if len(channels) != 2 or len(set(channels)) != 2 or not set(channels) <= {"ggf", "ttbar", "dihiggs"}:
        raise ValueError("Use two distinct supported channels in configs/pairwise_base.json.")
    tag = "phase1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    destination = Path(output).resolve() if output else project / "deployment" / tag
    destination.mkdir(parents=True, exist_ok=False)
    run_id = f"five_physics_seed{config['training']['seed']}"
    raw_name = "colliderml-data-pairwise-2500.tar.gz"
    raw_path = "/staging/k/kli398/" + raw_name
    image_path = "/staging/k/kli398/hep_ssl.sif"
    staged_dir = "/staging/k/kli398/" + tag
    prepared_url = "osdf:///chtc" + staged_dir + "/prepared-data.tar.gz"
    info = {"tag": tag, "run_id": run_id, "remote_relative": "hep_ssl_chtc/" + tag,
            "remote_directory": "/home/kli398/hep_ssl_chtc/" + tag,
            "stage_directory": staged_dir, "prepared_url": prepared_url,
            "raw_archive": raw_name, "raw_path": raw_path, "image_path": image_path,
            "access_login": "kli398@ap2002.chtc.wisc.edu",
            "transfer_login": "kli398@transfer.chtc.wisc.edu"}
    write_json(destination / "pairwise_chtc.json", config)
    write_json(destination / "deployment.json", info)
    write_json(destination / "input_paths.json", {})
    for name in ("run_prepare.sh", "prepare_data.py", "run_gpu.sh", "gpu_worker.py", "status.py", "README_ZH.md"):
        shutil.copy2(kit / name, destination / name)
    for p in destination.glob("*.sh"):
        p.chmod(0o755)
    excludes = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
    def archive_filter(member):
        parts = Path(member.name).parts
        if any(p in excludes for p in parts) or member.name.startswith("src/figures"):
            return None
        if member.name.endswith((".pyc", ".pyo")):
            return None
        return member
    with tarfile.open(destination / "hep_ssl-code.tar.gz", "w:gz") as out:
        for folder in ("src", "configs", "tests", "docs"):
            if (project / folder).exists():
                out.add(project / folder, arcname=folder, filter=archive_filter)
    # Keep small code/config in /home; the existing raw cache and prepared payload use staging.
    common = f'''universe = vanilla
container_image = osdf:///chtc{image_path}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
requirements = (TARGET.HasCHTCStaging == true)
request_cpus = 2
request_memory = 64GB
environment = "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1"
'''
    prepare = common + f'''# Run from the new /home submission directory, not /staging.
executable = run_prepare.sh
arguments = {raw_name}
transfer_input_files = hep_ssl-code.tar.gz, pairwise_chtc.json, prepare_data.py, input_paths.json, osdf:///chtc{raw_path}
transfer_output_files = prepared-data.tar.gz, prepare_status.json, prepare_details.json
transfer_output_remaps = "prepared-data.tar.gz = {prepared_url}"
request_disk = 80GB
log = prepare_$(Cluster)_$(Process).log
output = prepare_$(Cluster)_$(Process).out
error = prepare_$(Cluster)_$(Process).err
queue 1
'''
    (destination / "01_prepare.sub").write_text(prepare)
    gpu_common = common + '''executable = run_gpu.sh
request_gpus = 1
+WantGPULab = true
request_disk = 60GB
'''
    first = gpu_common + f'''+GPUJobLength = "short"
arguments = first {run_id} first_epoch.tar.gz
transfer_input_files = hep_ssl-code.tar.gz, pairwise_chtc.json, gpu_worker.py, {prepared_url}
transfer_output_files = first_epoch.tar.gz, first_status.json
log = first_$(Cluster)_$(Process).log
output = first_$(Cluster)_$(Process).out
error = first_$(Cluster)_$(Process).err
queue 1
'''
    (destination / "02_first_epoch.sub").write_text(first)
    resume = gpu_common + f'''+GPUJobLength = "medium"
arguments = continue {run_id} result_training.tar.gz
transfer_input_files = hep_ssl-code.tar.gz, gpu_worker.py, first_epoch.tar.gz, {prepared_url}
transfer_output_files = result_training.tar.gz, continue_status.json
log = continue_$(Cluster)_$(Process).log
output = continue_$(Cluster)_$(Process).out
error = continue_$(Cluster)_$(Process).err
queue 1
'''
    (destination / "03_continue.sub").write_text(resume)
    source = destination / "hep_ssl-code.tar.gz"
    if source.stat().st_size >= 1_000_000_000:
        raise ValueError("Code archive exceeds 1 GB. Exclude large non-code artifacts before this upload workflow.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (destination / "upload_sha256.txt").write_text(f"{digest}  hep_ssl-code.tar.gz\n")
    (destination / "location.env").write_text(
        f"DEPLOYMENT_REL='{info['remote_relative']}'\nSTAGE_DIR='{staged_dir}'\n")
    return destination


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--kit", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--output")
    a = ap.parse_args()
    print(build(a.project, a.kit, a.output))
