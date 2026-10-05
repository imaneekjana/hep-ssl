#!/usr/bin/env python3
"""Audit completed runs and write ONE batch of existing frozen-evaluation jobs.

Run this on the CHTC access point. Standard library only; no SSH, no submission,
no model changes and no new prepared data. The worker reuses the training code tar.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_fingerprint(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def token(value):
    value = str(value)
    if not value or re.search(r'[\s,;"\x00]', value):
        raise ValueError(f'Unsupported whitespace or delimiter in Condor value: {value!r}')
    return value


def local_file(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing local input or path outside deployment: {path}')
    return path


def archive_json(archive, name):
    f = archive.extractfile(name)
    if f is None:
        raise ValueError(f'Missing JSON member: {name}')
    with f:
        return json.load(f)


def inspect_pretraining(root):
    """Read small status/history/config records; actual checkpoint loading is in jobs."""
    root = Path(root).expanduser().resolve()
    info = read_json(root / 'deployment.json')
    manifest = read_json(root / 'manifest.json')
    registry = read_json(root / 'prepared_registry.json')
    if len(manifest) != 51 or len({r['run_id'] for r in manifest}) != 51:
        raise ValueError('Expected the original manifest with 51 unique runs.')
    source = local_file(root, 'hep_ssl-code.tar.gz')
    if sha256(source) != info['source_sha256']:
        raise ValueError('Original source archive differs from deployment.json.')
    with tarfile.open(source, 'r:gz') as tf:
        for name in ('src/evaluate_pairwise.py', 'src/evaluation/representations.py',
                     'chtc_phase1_steps/runtime_utils.py'):
            if not tf.getmember(name).isfile():
                raise ValueError(f'Source archive lacks {name}')

    latest = {}
    for path in sorted((root / 'attempts').glob('train_*/submission.json')):
        for row in read_json(path)['runs']:
            latest[row['run_id']] = path.parent

    receipts = {}
    for pair, record in registry.items():
        path = local_file(root, record['receipt_path'])
        receipt = read_json(path)
        if receipt.get('success') is not True or receipt.get('synthetic') is not False:
            raise ValueError(f'{pair}: no successful real-data preparation receipt.')
        if receipt.get('pair_id') != pair:
            raise ValueError(f'{pair}: receipt belongs to another pair.')
        fingerprint = json_fingerprint(receipt['preprocessing'])
        if receipt.get('prepared_fingerprint', receipt.get('metadata_fingerprint')) != fingerprint:
            raise ValueError(f'{pair}: preparation receipt fingerprint mismatch.')
        if receipt['preprocessing'].get('synthetic') is not False:
            raise ValueError(f'{pair}: synthetic prepared data is not allowed.')
        url = record['prepared_url']
        if url.rsplit('/', 1)[-1] != record['prepared_name']:
            raise ValueError(f'{pair}: prepared URL and basename disagree.')
        receipts[pair] = (path, receipt, fingerprint)

    selected, errors = [], []
    for row in manifest:
        run = row['run_id']
        try:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run):
                raise ValueError('invalid run name')
            if run not in latest:
                raise ValueError('no submitted training attempt')
            attempt = latest[run]
            status = read_json(attempt / f'status_{run}.json')
            if status.get('success') is not True or status.get('exit_code') != 0:
                raise ValueError('latest attempt did not finish successfully')
            if status.get('archive_exit_code', 0) != 0:
                raise ValueError('result archiving failed')
            if status.get('completed_epochs') != 18:
                raise ValueError(f"only {status.get('completed_epochs')} completed epochs")
            pair = row['prepared_id']
            receipt_path, receipt, fingerprint = receipts[pair]
            if status.get('prepared_fingerprint') != fingerprint:
                raise ValueError('training and prepared data fingerprints disagree')
            training_source = status.get('details', {}).get('code_archive_sha256')
            if training_source is not None and training_source != info['source_sha256']:
                raise ValueError('training used another source archive')
            result = local_file(root, str((attempt / f'result_{run}.tar.gz').relative_to(root)))
            with tarfile.open(result, 'r:gz') as tf:
                config = archive_json(tf, f'{run}/config.json')
                history = archive_json(tf, f'{run}/history.json')
                for name in ('best.pt', 'last.pt'):
                    member = tf.getmember(f'{run}/checkpoints/{name}')
                    if not member.isfile() or member.size <= 0:
                        raise ValueError(f'missing/non-file/empty {name}')
            if config['mode'] != 'five_anisotropic_physics':
                raise ValueError('unexpected model mode')
            if config['training']['epochs'] != 18:
                raise ValueError('unexpected configured epoch count')
            if config['data']['channels'] != [row['channel_a'], row['channel_b']]:
                raise ValueError('checkpoint configuration has another channel pair')
            if config['augmentation']['order'] != row['augmentation_order']:
                raise ValueError('augmentation configuration differs from manifest')
            if [h['epoch'] for h in history] != list(range(18)):
                raise ValueError('history does not contain epochs 0..17 in order')
            if not all(math.isfinite(h[split]['loss']) for h in history for split in ('train', 'val')):
                raise ValueError('nonfinite epoch loss')
            best_row = min(history, key=lambda h: h['val']['loss'])
            record = registry[pair]
            selected.append({
                'run_id': run, 'pair_id': pair, 'result_path': str(result),
                'result_name': result.name, 'prepared_url': record['prepared_url'],
                'prepared_name': record['prepared_name'], 'receipt_path': str(receipt_path),
                'receipt_name': receipt_path.name, 'prepared_fingerprint': fingerprint,
                'completed_epochs': 18, 'best_epoch_from_history': best_row['epoch'] + 1,
                'best_validation_loss': best_row['val']['loss'],
                'final_train_loss': history[-1]['train']['loss'],
                'final_validation_loss': history[-1]['val']['loss'],
            })
        except (KeyError, OSError, ValueError, TypeError, tarfile.TarError) as exc:
            errors.append(f'{run}: {exc}')
    if errors:
        raise ValueError('Pretraining audit found incomplete/mismatched results:\n' + '\n'.join(errors))
    return info, selected


def build(root, device='cuda', output=None):
    root = Path(root).expanduser().resolve()
    if device not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    info, rows = inspect_pretraining(root)
    print('PRETRAINING_AUDIT_OK: 51/51 latest runs; 18 epochs each; histories/checkpoint files present.')
    output = (Path(output).expanduser().resolve() if output else
              root / 'classifiers' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'results'):
        (output / name).mkdir()
    wrapper = output / 'run_classifier.sh'
    shutil.copy2(Path(__file__).with_name('run_classifier.sh'), wrapper)
    wrapper.chmod(0o755)
    # Absolute submission-side paths; HTCondor transfers file basenames into each job.
    source = root / 'hep_ssl-code.tar.gz'
    fields = ('run_id', 'result_path', 'result_name', 'prepared_url', 'prepared_name',
              'receipt_path', 'receipt_name')
    with (output / 'classifiers.tsv').open('w') as f:
        for row in rows:
            f.write('\t'.join(token(row[k]) for k in fields) + '\n')
    audit_fields = ('run_id', 'pair_id', 'completed_epochs', 'best_epoch_from_history',
                    'best_validation_loss', 'final_train_loss', 'final_validation_loss',
                    'prepared_fingerprint', 'result_path')
    with (output / 'pretraining_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in audit_fields} for row in rows)
    settings = info['settings']
    image = settings['container_image']
    if not image.startswith(('osdf:', 'file:', 'docker:')):
        image = ('file://' if settings['transfer_protocol'] == 'file' else 'osdf:///chtc') + image
    cpus = int(settings.get('request_cpus', 2))
    gpu = ('request_gpus = 1\n+WantGPULab = true\n+GPUJobLength = "medium"\n'
           if device == 'cuda' else '')
    submit = f'''# Frozen evaluation of the completed 51 pretraining runs. Submit here once.
universe = vanilla
container_image = {token(image)}
executable = run_classifier.sh
arguments = $(eval_run) $(eval_prepared_name) $(eval_result_name) $(eval_receipt_name) {device}
transfer_input_files = {token(source)}, $(eval_result_path), $(eval_prepared_url), $(eval_receipt_path)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = classifier_$(eval_run).tar.gz, classifier_status_$(eval_run).json
transfer_output_remaps = "classifier_$(eval_run).tar.gz = results/classifier_$(eval_run).tar.gz; classifier_status_$(eval_run).json = results/classifier_status_$(eval_run).json"
requirements = (TARGET.HasCHTCStaging == true)
request_cpus = {cpus}
request_memory = {token(settings.get('request_memory', '64GB'))}
request_disk = {token(settings.get('train_disk', '60GB'))}
environment = "OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1"
{gpu}log = logs/classifier_$(eval_run)_$(Cluster)_$(Process).log
output = logs/classifier_$(eval_run)_$(Cluster)_$(Process).out
error = logs/classifier_$(eval_run)_$(Cluster)_$(Process).err
queue eval_run,eval_result_path,eval_result_name,eval_prepared_url,eval_prepared_name,eval_receipt_path,eval_receipt_name from classifiers.tsv
'''
    (output / 'classifiers.sub').write_text(submit)
    (output / 'evaluation_plan.json').write_text(json.dumps({
        'pretraining_deployment': str(root), 'source_archive': str(source),
        'source_archive_sha256': info['source_sha256'], 'device': device,
        'jobs': 51, 'encoder_mode': 'pretrained', 'checkpoint': 'best.pt',
        'classifier_c': 1.0, 'probe_alpha': 1.0, 'seed': 42, 'clean_inputs': True,
        'model_frozen': True, 'spaces': ['general', 'energy', 'eta', 'phi', 'local', 'concat'],
        'existing_evaluator_also_runs_physics_probes_and_simple_baselines': True,
        'runs': rows,
    }, indent=2) + '\n')
    print(f'CLASSIFIER_DIRECTORY={output}')
    print('No jobs submitted. In that directory, run: condor_submit classifiers.sub')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    try:
        build(args.deployment, args.device)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f'NOT_READY: {exc}\n')


if __name__ == '__main__':
    main()
