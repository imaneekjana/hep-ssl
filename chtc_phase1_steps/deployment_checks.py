"""Small, dependency-free preparation gate. Heavy validation stays on execution nodes."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

try:
    from .runtime_utils import sha256_file, metadata_fingerprint, write_json
except ImportError:
    from runtime_utils import sha256_file, metadata_fingerprint, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def local_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Deployment path escapes its directory: {relative}")
    return path


def verify_deployment(root):
    root = Path(root)
    info = read_json(root / "deployment.json")
    if sha256_file(root / "hep_ssl-code.tar.gz") != info["source_sha256"]:
        raise ValueError("Source archive changed; generate a new deployment")
    if sha256_file(root / "manifest.json") != info["manifest_sha256"]:
        raise ValueError("Manifest changed; generate a new deployment")
    manifest = read_json(root / "manifest.json")
    if len(manifest) != 51 or len({r['run_id'] for r in manifest}) != 51:
        raise ValueError("Expected 51 unique runs")
    for row in manifest:
        if sha256_file(local_path(root, row['config_path'])) != row['config_sha256']:
            raise ValueError(f"Config changed: {row['run_id']}; generate a new deployment")
    return info, manifest


def completed_job(job_id, *, allow_removed=False):
    """History only contains finished jobs; absent/running/held is never ready."""
    result = subprocess.run(["condor_history", str(job_id), "-json", "-attributes",
                             "ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal"],
                            check=True, capture_output=True, text=True)
    cluster, proc = map(int, job_id.split('.'))
    ads = [a for a in json.loads(result.stdout)
           if a.get('ClusterId') == cluster and a.get('ProcId') == proc]
    if len(ads) != 1 or ads[0].get('JobStatus') not in ({3, 4} if allow_removed else {4}):
        raise RuntimeError(f"Job {job_id} has no normal completed history record; inspect condor_q/condor_history")
    if not allow_removed and ads[0].get('ExitBySignal', False):
        raise RuntimeError(f"Job {job_id} terminated by signal")
    return ads[0]


def validate_record(root, pair_id, record, config, *, query_history=True):
    status_path = local_path(root, record['status_path'])
    receipt_path = local_path(root, record['receipt_path'])
    if not status_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"{pair_id}: preparation outputs have not returned")
    status, receipt = read_json(status_path), read_json(receipt_path)
    if status.get('success') is not True or status.get('exit_code') != 0:
        raise RuntimeError(f"{pair_id}: preparation failed; inspect {status_path}")
    if receipt.get('success') is not True or receipt.get('synthetic') is not False:
        raise ValueError(f"{pair_id}: receipt does not confirm real prepared data")
    if status.get('receipt_sha256') != sha256_file(receipt_path):
        raise ValueError(f"{pair_id}: status/receipt digest mismatch")
    for obj in (receipt, status):
        if obj.get('pair_id') != pair_id or obj.get('archive') != record['prepared_name']:
            raise ValueError(f"{pair_id}: wrong preparation identity/archive")
        if obj.get('config_sha256') != record['config_sha256']:
            raise ValueError(f"{pair_id}: preparation configuration digest mismatch")
        if obj.get('code_archive_sha256') != record['source_sha256']:
            raise ValueError(f"{pair_id}: preparation source digest mismatch")
    if status.get('archive_sha256') != receipt.get('archive_sha256'):
        raise ValueError(f"{pair_id}: archive digest mismatch")
    if not isinstance(receipt.get('archive_sha256'), str) or len(receipt['archive_sha256']) != 64:
        raise ValueError(f"{pair_id}: missing archive digest")
    metadata = receipt['preprocessing']
    if metadata.get('synthetic') is not False:
        raise ValueError(f"{pair_id}: synthetic data is forbidden")
    if metadata_fingerprint(metadata) != receipt['metadata_fingerprint']:
        raise ValueError(f"{pair_id}: prepared fingerprint mismatch")
    checksum_metadata = dict(metadata)
    recorded_hash = checksum_metadata.pop('metadata_hash')
    if metadata_fingerprint(checksum_metadata) != recorded_hash:
        raise ValueError(f"{pair_id}: prepared metadata checksum mismatch")
    if metadata_fingerprint(metadata['manifest']) != metadata['manifest_hash']:
        raise ValueError(f"{pair_id}: split manifest checksum mismatch")
    for name in ('metadata_hash', 'manifest_hash'):
        if receipt.get(name) != metadata[name]:
            raise ValueError(f"{pair_id}: receipt {name} differs from preprocessing")
    if receipt.get('input_paths_sha256') != record.get('input_paths_sha256'):
        raise ValueError(f"{pair_id}: preparation input-path digest mismatch")
    preparation = metadata['preparation_config']
    for section in ('data', 'grid', 'targets'):
        for key, value in config[section].items():
            if value is not None and value != preparation[section].get(key):
                raise ValueError(f"{pair_id}: incompatible prepared {section}.{key}")
    if preparation['data']['channels'] != pair_id.split('_'):
        raise ValueError(f"{pair_id}: wrong prepared channels")
    if type(receipt['archive_bytes']) is not int or receipt['archive_bytes'] <= 0:
        raise ValueError(f"{pair_id}: empty archive")
    if status.get('archive_bytes') != receipt['archive_bytes']:
        raise ValueError(f"{pair_id}: archive size mismatch")
    if receipt['archive_bytes'] > 30_000_000_000 and record['prepared_url'].startswith('osdf:'):
        raise ValueError(f"{pair_id}: archive exceeds 30 GB; use file:// staged transfer")
    completion = record.get('completion')
    if completion is None:
        if not query_history:
            raise RuntimeError(f"{pair_id}: preparation has not passed the scheduler completion gate")
        completion = completed_job(record['job_id'])
        if completion.get('ExitCode') != 0:
            raise RuntimeError(f"{pair_id}: scheduler exit was nonzero")
        record['completion'] = completion
    if completion.get('JobStatus') != 4 or completion.get('ExitCode') != 0 or completion.get('ExitBySignal', False):
        raise RuntimeError(f"{pair_id}: preparation job did not finish successfully")
    cluster, proc = map(int, record['job_id'].split('.'))
    if completion.get('ClusterId') != cluster or completion.get('ProcId') != proc:
        raise ValueError(f"{pair_id}: scheduler completion belongs to another job")
    record['metadata_fingerprint'] = receipt['metadata_fingerprint']
    record['archive_sha256'] = receipt['archive_sha256']
    return receipt


def check_prepared(root, *, query_history=True):
    root = Path(root)
    info, _ = verify_deployment(root)
    registry = read_json(root / 'prepared_registry.json')
    raw_hashes, revisions = set(), set()
    for pair_id, pair in info['pairs'].items():
        if pair_id not in registry:
            raise RuntimeError(f"{pair_id}: no preparation submitted or reused")
        config = read_json(local_path(root, pair['config_path']))
        receipt = validate_record(root, pair_id, registry[pair_id], config, query_history=query_history)
        if receipt.get('raw_archive_sha256'):
            raw_hashes.add(receipt['raw_archive_sha256'])
        revisions.add(receipt['preprocessing']['data']['dataset_revision'])
    if len(raw_hashes) > 1 or len(revisions) != 1:
        raise ValueError('The three prepared tasks used different raw archive snapshots')
    write_json(root / 'prepared_registry.json', registry)
    return registry


def reuse_prepared(destination, previous):
    """Copy small verified receipts; keep the immutable staged archive URL unchanged."""
    destination, previous = Path(destination), Path(previous)
    info, _ = verify_deployment(destination)
    old_info, _ = verify_deployment(previous)
    if info['preparation_source_sha256'] != old_info['preparation_source_sha256']:
        raise ValueError('Preprocessing source changed; previous deployment cannot be reused automatically')
    old_registry = read_json(previous / 'prepared_registry.json')
    registry = {}
    (destination / 'reused').mkdir()
    for pair_id, record in old_registry.items():
        if pair_id not in info['pairs']:
            continue
        config = read_json(destination / info['pairs'][pair_id]['config_path'])
        record = copy.deepcopy(record)
        validate_record(previous, pair_id, record, config, query_history=False)
        for field in ('status_path', 'receipt_path'):
            source = local_path(previous, record[field])
            target = destination / 'reused' / source.name
            shutil.copy2(source, target)
            record[field] = str(target.relative_to(destination))
        registry[pair_id] = record
    write_json(destination / 'prepared_registry.json', registry)
    return registry
