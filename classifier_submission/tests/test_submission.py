"""Synthetic metadata fixtures only: no real checkpoints or cluster submission."""
import copy
import csv
import io
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_classifiers as module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_tar(path, entries):
    with tarfile.open(path, 'w:gz') as tf:
        for name, value in entries.items():
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            tf.addfile(item, io.BytesIO(payload))


@pytest.fixture
def deployment(tmp_path):
    root = tmp_path / 'pretraining'
    root.mkdir()
    source = root / 'hep_ssl-code.tar.gz'
    write_tar(source, {name: b'# synthetic interface fixture\n' for name in (
        'src/evaluate_pairwise.py', 'src/evaluation/representations.py',
        'chtc_phase1_steps/runtime_utils.py')})
    info = {'source_sha256': module.sha256(source), 'settings': {
        'container_image': '/staging/k/kli398/hep_ssl.sif', 'transfer_protocol': 'osdf',
        'request_cpus': 2, 'request_memory': '64GB', 'train_disk': '60GB'}}
    write_json(root / 'deployment.json', info)
    latest = root / 'attempts/train_20261002_010000'
    registry, manifest = {}, []
    transforms = ('rotate', 'energy_noise', 'xyz_noise', 'shift', 'crop')
    abbreviations = dict(zip(transforms, 'rexsc'))
    subsets = [()] + [c for n in (3, 4, 5) for c in combinations(transforms, n)]
    for a, b in (('ggf', 'ttbar'), ('ggf', 'dihiggs'), ('ttbar', 'dihiggs')):
        pair = f'{a}_{b}'
        metadata = {'synthetic': False, 'fixture_only': True, 'channels': [a, b]}
        fingerprint = module.json_fingerprint(metadata)
        receipt_name = f'prepare_{pair}_details.json'
        receipt_path = root / 'attempts/prepare_test' / receipt_name
        write_json(receipt_path, {'success': True, 'synthetic': False, 'pair_id': pair,
                                 'preprocessing': metadata, 'prepared_fingerprint': fingerprint,
                                 'archive_sha256': '0' * 64})
        registry[pair] = {'receipt_path': str(receipt_path.relative_to(root)),
                          'prepared_name': f'prepared_{pair}.tar.gz',
                          'prepared_url': f'osdf:///chtc/staging/k/kli398/prepared_{pair}.tar.gz'}
        for subset in subsets:
            run = pair + '_' + (''.join(abbreviations[t] for t in subset) or 'none')
            manifest.append({'run_id': run, 'channel_a': a, 'channel_b': b,
                             'augmentation_order': list(subset), 'prepared_id': pair})
            cfg = {'mode': 'five_anisotropic_physics', 'training': {'epochs': 18},
                   'data': {'channels': [a, b]}, 'augmentation': {'order': list(subset)}}
            hist = [{'epoch': n, 'train': {'loss': 10 - n / 10},
                     'val': {'loss': 12 - n / 10}} for n in range(18)]
            write_json(latest / f'status_{run}.json', {
                'success': True, 'exit_code': 0, 'archive_exit_code': 0,
                'completed_epochs': 18, 'prepared_fingerprint': fingerprint,
                'details': {'code_archive_sha256': info['source_sha256']}})
            # These bytes only test presence, never torch.load or a training claim.
            write_tar(latest / f'result_{run}.tar.gz', {
                f'{run}/config.json': cfg, f'{run}/history.json': hist,
                f'{run}/checkpoints/best.pt': b'not-a-real-checkpoint',
                f'{run}/checkpoints/last.pt': b'not-a-real-checkpoint'})
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'prepared_registry.json', registry)
    write_json(latest / 'submission.json', {'runs': [{'run_id': r['run_id']} for r in manifest]})
    return root


def alter_status(root, **changes):
    path = root / 'attempts/train_20261002_010000/status_ggf_ttbar_none.json'
    data = module.read_json(path)
    data.update(changes)
    write_json(path, data)


def alter_tar(root, transform):
    path = root / 'attempts/train_20261002_010000/result_ggf_ttbar_none.tar.gz'
    with tarfile.open(path) as tf:
        entries = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}
    transform(entries)
    write_tar(path, entries)


def test_build_51_matching_jobs_and_six_spaces(deployment):
    destination = module.build(deployment)
    rows = list(csv.reader((destination / 'classifiers.tsv').open(), delimiter='\t'))
    assert len(rows) == 51 and len({r[0] for r in rows}) == 51
    for run, result, result_name, url, prepared_name, receipt, receipt_name in rows:
        assert Path(result).name == result_name == f'result_{run}.tar.gz'
        assert url.endswith('/' + prepared_name)
        assert Path(receipt).name == receipt_name
        assert module.read_json(receipt)['pair_id'] == '_'.join(run.split('_')[:2])
    plan = module.read_json(destination / 'evaluation_plan.json')
    assert plan['spaces'] == ['general', 'energy', 'eta', 'phi', 'local', 'concat']
    assert plan['checkpoint'] == 'best.pt' and plan['encoder_mode'] == 'pretrained'
    text = (destination / 'classifiers.sub').read_text()
    assert 'manifest =' not in text and '-append' not in text
    assert 'request_gpus = 1' in text
    assert text.count('\nqueue ') == 1 and ' from classifiers.tsv' in text
    assert len(list(csv.DictReader((destination / 'pretraining_audit.csv').open()))) == 51
    assert not (deployment / 'prepared').exists()


def test_cpu_option_no_gpu_request(deployment):
    destination = module.build(deployment, device='cpu')
    text = (destination / 'classifiers.sub').read_text()
    assert 'request_gpus' not in text and '$(eval_receipt_name) cpu' in text


@pytest.mark.parametrize('changes,match', [
    ({'success': False, 'exit_code': 1}, 'did not finish'),
    ({'completed_epochs': 1}, 'only 1'),
    ({'archive_exit_code': 1}, 'archiving failed'),
    ({'prepared_fingerprint': 'wrong'}, 'fingerprints disagree'),
])
def test_status_failures_block_generation(deployment, changes, match):
    alter_status(deployment, **changes)
    with pytest.raises(ValueError, match=match):
        module.build(deployment)
    assert not (deployment / 'classifiers').exists()


def test_latest_failed_attempt_not_silently_replaced(deployment):
    root = deployment / 'attempts/train_20261003_010000'
    write_json(root / 'submission.json', {'runs': [{'run_id': 'ggf_ttbar_none'}]})
    write_json(root / 'status_ggf_ttbar_none.json', {'success': False, 'exit_code': 1})
    with pytest.raises(ValueError, match='did not finish'):
        module.inspect_pretraining(deployment)


def test_history_incomplete(deployment):
    def change(entries):
        key = 'ggf_ttbar_none/history.json'
        entries[key] = json.dumps(json.loads(entries[key])[:-1]).encode()
    alter_tar(deployment, change)
    with pytest.raises(ValueError, match='epochs 0..17'):
        module.inspect_pretraining(deployment)


def test_nonfinite_history(deployment):
    def change(entries):
        key = 'ggf_ttbar_none/history.json'
        hist = json.loads(entries[key]); hist[4]['val']['loss'] = float('nan')
        entries[key] = json.dumps(hist).encode()
    alter_tar(deployment, change)
    with pytest.raises(ValueError, match='nonfinite'):
        module.inspect_pretraining(deployment)


def test_missing_checkpoint(deployment):
    alter_tar(deployment, lambda e: e.pop('ggf_ttbar_none/checkpoints/best.pt'))
    with pytest.raises(ValueError, match='best.pt'):
        module.inspect_pretraining(deployment)


def test_does_not_change_inputs(deployment):
    before = {str(p): module.sha256(p) for p in deployment.rglob('*') if p.is_file()}
    module.build(deployment)
    assert all(module.sha256(p) == digest for p, digest in before.items())


def test_worker_shell_syntax():
    path = Path(module.__file__).with_name('run_classifier.sh')
    subprocess.run(['bash', '-n', str(path)], check=True)


def test_worker_preserves_failure_log_without_inputs(tmp_path):
    path = Path(module.__file__).with_name('run_classifier.sh')
    run = 'ggf_ttbar_none'
    result = subprocess.run(['bash', str(path), run, 'prepared.tar.gz',
                             f'result_{run}.tar.gz', 'receipt.json', 'cpu'],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    status = module.read_json(tmp_path / f'classifier_status_{run}.json')
    assert status['success'] is False
    assert (tmp_path / f'classifier_{run}.tar.gz').is_file()


def test_classifier_is_frozen_clean_best_and_existing_entry():
    text = Path(module.__file__).with_name('run_classifier.sh').read_text()
    assert '-m src.evaluate_pairwise' in text
    assert '--encoder-mode pretrained' in text
    assert 'checkpoints/best.pt' in text
    assert '--classifier-c 1.0 --probe-alpha 1.0 --seed 42' in text
    assert 'src.train_pairwise' not in text
