"""Execution-node validation contracts, with no submitted jobs or CUDA simulation.

The generated hits and minimal checkpoint below are parser fixtures only.
No model training or successful GPU execution is claimed by these tests.
"""
import copy
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest
import torch

from chtc_phase1_steps import gpu_worker
from chtc_phase1_steps.runtime_utils import metadata_fingerprint, sha256_file
from src.config import load_config
from src.prepare_pairwise import prepare, synthetic_events


@pytest.fixture(scope='module')
def receipt_fixture(tmp_path_factory):
    base = tmp_path_factory.mktemp('gpu_validation_fixture')
    config = load_config()
    config['training']['device'] = 'cuda'
    config['data'].update(dataset_revision='unit-test-fixture-v1', events_per_channel=5)
    config['grid'].update(n_eta=8, n_phi=16)
    events = synthetic_events(config['data']['channels'], 5)
    for event in events:
        event.dataset_revision = 'unit-test-fixture-v1'
    prepared = prepare(config, base / 'artifact' / 'prepared', events=events)
    archive = base / 'prepared_ggf_ttbar.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(prepared.directory, arcname='prepared')
    receipt = {'pair_id': 'ggf_ttbar', 'success': True, 'synthetic': False,
               'archive_sha256': sha256_file(archive), 'preprocessing': prepared.metadata,
               'prepared_fingerprint': metadata_fingerprint(prepared.metadata)}
    saved_config = gpu_worker.resolved_config(config, prepared.metadata)
    saved_config['prepared_dir'] = '/previous_execution/scratch/prepared'
    # Only the checkpoint metadata parser is exercised; no fake model weights
    # are ever passed to train(), and cuda_probe() is never patched to succeed.
    state = {'schema_version': 1, 'config': saved_config, 'model_state': {}, 'objective_state': {},
             'optimizer_state': {}, 'scheduler_state': {}, 'scaler_state': {}, 'epoch': 0,
             'global_step': 1, 'best_validation': 1, 'history': [], 'rng_state': {},
             'preprocessing': prepared.metadata, 'prepared_fingerprint': receipt['prepared_fingerprint']}
    run = base / 'prior' / 'test_run'
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps(saved_config))
    torch.save(state, run / 'checkpoints' / 'last.pt')
    resume_archive = base / 'resume_input.tar.gz'
    with tarfile.open(resume_archive, 'w:gz') as stream:
        stream.add(run, arcname='test_run')
    return config, receipt, archive, resume_archive


def stage_inputs(directory, fixture, config=None, receipt=None, resume=False):
    original_config, original_receipt, archive, resume_archive = fixture
    shutil.copy2(archive, directory / archive.name)
    if resume:
        shutil.copy2(resume_archive, directory / resume_archive.name)
    (directory / 'receipt.json').write_text(json.dumps(receipt or original_receipt))
    (directory / 'config.json').write_text(json.dumps(config or original_config))
    return SimpleNamespace(config='config.json', run_id='test_run', receipt='receipt.json',
                           prepared_archive=archive.name, stop_after_epoch=0,
                           resume_archive=resume_archive.name if resume else '-')


@pytest.mark.parametrize('case', [
    'valid', 'sha_mismatch', 'wrong_pair', 'synthetic', 'fingerprint_mismatch', 'explicit_grid_mismatch',
])
def test_prepared_receipt_validation_before_cuda(tmp_path, monkeypatch, receipt_fixture, case):
    config, receipt = copy.deepcopy(receipt_fixture[:2])
    if case == 'sha_mismatch':
        receipt['archive_sha256'] = '0' * 64
    elif case == 'wrong_pair':
        receipt['pair_id'] = 'ggf_dihiggs'
    elif case == 'synthetic':
        receipt['synthetic'] = True
    elif case == 'fingerprint_mismatch':
        receipt['prepared_fingerprint'] = '0' * 64
    elif case == 'explicit_grid_mismatch':
        config['grid']['n_eta'] = 16
    args = stage_inputs(tmp_path, receipt_fixture, config, receipt)
    monkeypatch.chdir(tmp_path)
    details = {}
    if case == 'valid':
        _, _, checkpoint = gpu_worker.verify_inputs(args, details)
        assert checkpoint is None
        assert details['prepared_fingerprint'] == receipt['prepared_fingerprint']
        assert not Path('outputs').exists(), 'validation must not precreate the fresh run directory'
    else:
        with pytest.raises(ValueError):
            gpu_worker.verify_inputs(args, details)


@pytest.mark.parametrize('case', [
    'valid_relocation', 'augmentation_changed', 'optimizer_changed', 'horizon_changed', 'seed_changed',
])
def test_resume_allows_only_resolved_fields_and_path_relocation(tmp_path, monkeypatch, receipt_fixture, case):
    config = copy.deepcopy(receipt_fixture[0])
    if case == 'augmentation_changed':
        config['augmentation']['order'] = ['rotate']
    elif case == 'optimizer_changed':
        config['training']['lr'] *= 2
    elif case == 'horizon_changed':
        config['training']['epochs'] += 1
    elif case == 'seed_changed':
        config['training']['seed'] += 1
    args = stage_inputs(tmp_path, receipt_fixture, config=config, resume=True)
    monkeypatch.chdir(tmp_path)
    details = {}
    if case == 'valid_relocation':
        _, _, checkpoint = gpu_worker.verify_inputs(args, details)
        assert details['completed_epochs'] == 1
        assert checkpoint == Path('outputs/test_run/checkpoints/last.pt')
        assert checkpoint.is_file()
    else:
        with pytest.raises(ValueError, match='Resume configuration differs'):
            gpu_worker.verify_inputs(args, details)


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only rejection test requires actual CUDA absence')
def test_real_cuda_probe_has_no_cpu_fallback():
    pytest.importorskip('torch_cluster')
    with pytest.raises(RuntimeError, match='CUDA is unavailable. No CPU fallback'):
        gpu_worker.cuda_probe()
