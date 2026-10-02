"""Real preparation entry point on tiny local Parquet fixtures, without CHTC.

The records are manufactured test inputs; passing these checks does not claim
that ColliderML production data or a cluster job has been run successfully.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from chtc_phase1_steps.runtime_utils import extract_safe_archive, metadata_fingerprint
from src.config import load_config

PROJECT = Path(__file__).resolve().parents[1]
KIT = PROJECT / 'chtc_phase1_steps'
PAIR = 'ggf_ttbar'
CONFIG = PAIR + '_none.json'
ARCHIVE = 'prepared_ggf_ttbar_fixture.tar.gz'
RECEIPT = 'prepare_ggf_ttbar_details.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_fixture(tmp_path, scenario):
    pl = pytest.importorskip('polars')
    scratch = tmp_path / 'execution node with spaces'
    scratch.mkdir()
    for name in ('prepare_data.py', 'run_prepare.sh', 'runtime_utils.py'):
        shutil.copy2(KIT / name, scratch / name)
    config = load_config(PROJECT / 'configs/pairwise_base.json')
    config['data']['channels'] = ['ggf', 'ttbar']
    config['data']['dataset_revision'] = None
    config['data']['events_per_channel'] = 5
    config['grid'].update(n_eta=8, n_phi=16)
    config['augmentation']['order'] = []
    (scratch / CONFIG).write_text(json.dumps(config))
    # All three overrides are valid configuration; the third pair is not read.
    (scratch / 'input_paths.json').write_text(json.dumps({
        'ggf': 'ggf_pu0_calo_hits', 'ttbar': 'ttbar_pu0_calo_hits',
        'dihiggs': 'nonexistent_unused_channel_override',
    }))
    with tarfile.open(scratch / 'hep_ssl-code.tar.gz', 'w:gz') as bundle:
        bundle.add(PROJECT / 'src', arcname='src',
                   filter=lambda member: None if '__pycache__' in member.name else member)
    fixture_root = tmp_path / 'fixture_inputs'
    fixture_root.mkdir()
    for channel in ('ggf', 'ttbar'):
        directory = fixture_root / f'{channel}_pu0_calo_hits'
        directory.mkdir()
        rows = []
        for event_id in range(5):
            eta, phi = np.linspace(-1., 1., 20), np.linspace(-3., 3., 20)
            row = {'event_id': event_id,
                   'x': (1000 * np.cos(phi)).tolist(), 'y': (1000 * np.sin(phi)).tolist(),
                   'z': (1000 * np.sinh(eta)).tolist(),
                   'total_energy': np.full(20, 1. + event_id).tolist()}
            if scenario == 'bad_schema':
                del row['event_id']
            rows.append(row)
        pl.DataFrame(rows).write_parquet(directory / 'train-0000.parquet')
    with tarfile.open(scratch / 'raw.tar.gz', 'w:gz') as bundle:
        for path in sorted(fixture_root.iterdir()):
            bundle.add(path, arcname=path.name)
        if scenario == 'unsafe_archive':
            link = tarfile.TarInfo('outside')
            link.type, link.linkname = tarfile.SYMTYPE, '/tmp'
            bundle.addfile(link)
    return scratch, config


def run_fixture(scratch):
    environment = os.environ.copy()
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        ['bash', 'run_prepare.sh', PAIR, CONFIG, 'raw.tar.gz', ARCHIVE, RECEIPT],
        cwd=scratch, env=environment, capture_output=True, text=True, timeout=90)
    status = json.loads((scratch / f'prepare_{PAIR}_status.json').read_text())
    receipt = json.loads((scratch / RECEIPT).read_text())
    assert status['pair_id'] == receipt['pair_id'] == PAIR
    assert status['archive'] == receipt['archive'] == ARCHIVE
    assert status['receipt_sha256'] == digest(scratch / RECEIPT)
    assert status['archive_sha256'] == receipt['archive_sha256'] == digest(scratch / ARCHIVE)
    assert status['archive_bytes'] == receipt['archive_bytes'] == (scratch / ARCHIVE).stat().st_size
    assert status['config_sha256'] == receipt['config_sha256'] == digest(scratch / CONFIG)
    assert status['code_archive_sha256'] == receipt['code_archive_sha256'] == digest(scratch / 'hep_ssl-code.tar.gz')
    return completed, status, receipt


def test_prepare_wrapper_real_input_path_success_has_complete_bound_receipt(tmp_path):
    scratch, config = execution_fixture(tmp_path, 'success')
    completed, status, receipt = run_fixture(scratch)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert status['success'] is receipt['success'] is True
    assert status['exit_code'] == 0
    assert receipt['synthetic'] is False
    assert receipt['events'] == 10
    assert receipt['raw_archive_sha256'] == digest(scratch / 'raw.tar.gz')
    assert receipt['dataset_revision'] == 'local-cache-sha256:' + receipt['raw_archive_sha256']
    assert receipt['source_archive_sha256'] == receipt['code_archive_sha256']
    assert set(receipt['selected_source_files']) == {'ggf', 'ttbar'}
    assert receipt['requested_compatibility'] == {key: config[key] for key in ('data', 'grid', 'targets')}
    metadata = receipt['preprocessing']
    assert metadata['synthetic'] is False
    assert receipt['prepared_fingerprint'] == receipt['metadata_fingerprint'] == metadata_fingerprint(metadata)
    assert receipt['metadata_hash'] == metadata['metadata_hash']
    assert receipt['manifest_hash'] == metadata['manifest_hash'] == metadata_fingerprint(metadata['manifest'])
    assert receipt['preparation_config'] == metadata['preparation_config']
    assert receipt['compatibility'] == {key: metadata['preparation_config'][key] for key in ('data', 'grid', 'targets')}
    assert set(metadata) >= {'feature_stats', 'summary_stats', 'target_stats', 'target_definitions'}
    with tarfile.open(scratch / ARCHIVE) as bundle:
        assert {'prepared/prepared.json', 'prepared/manifest.json', 'prepared/events.npz'} <= set(bundle.getnames())
        archived = json.load(bundle.extractfile('prepared/prepared.json'))
    assert archived == metadata
    assert not (scratch / (ARCHIVE + '.partial')).exists()


@pytest.mark.parametrize('scenario', ['bad_schema', 'unsafe_archive'])
def test_prepare_wrapper_failures_are_nonzero_and_return_diagnostics(tmp_path, scenario):
    scratch, _ = execution_fixture(tmp_path, scenario)
    completed, status, receipt = run_fixture(scratch)
    assert completed.returncode != 0
    assert status['success'] is receipt['success'] is False
    assert status['exit_code'] != 0
    assert receipt['error']
    with tarfile.open(scratch / ARCHIVE) as bundle:
        members = set(bundle.getnames())
        failed = f'failed_preparation_{PAIR}/FAILED.json'
        assert failed in members
        assert not any(name.startswith('prepared/') for name in members)
        diagnostic = json.load(bundle.extractfile(failed))
    assert diagnostic['success'] is False
    if scenario == 'unsafe_archive':
        assert 'links and special files are not allowed' in completed.stderr
        assert not (scratch / 'raw').exists()  # preflight rejects before extraction
    else:
        assert 'No IDs will be fabricated' in completed.stderr


def test_safe_extractor_rejects_destination_symlink(tmp_path):
    archive = tmp_path / 'empty.tar.gz'
    with tarfile.open(archive, 'w:gz'):
        pass
    outside = tmp_path / 'outside'
    outside.mkdir()
    linked = tmp_path / 'linked'
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='destination must not be a symlink'):
        extract_safe_archive(archive, linked)


@pytest.mark.parametrize('scenario', ['matching', 'explicit_revision', 'incompatible', 'tampered', 'synthetic'])
def test_verify_existing_archive_preserves_bytes_and_validates_metadata(tmp_path, scenario):
    scratch, config = execution_fixture(tmp_path, 'success')
    if scenario == 'explicit_revision':
        config['data']['dataset_revision'] = 'verified-upstream-fixture'
        (scratch / CONFIG).write_text(json.dumps(config))
    first, _, original_receipt = run_fixture(scratch)
    assert first.returncode == 0, first.stdout + first.stderr
    verifier = tmp_path / 'verification node'
    verifier.mkdir()
    for name in ('prepare_data.py', 'run_prepare.sh', 'runtime_utils.py', 'hep_ssl-code.tar.gz', CONFIG, ARCHIVE):
        shutil.copy2(scratch / name, verifier / name)
    if scenario == 'incompatible':
        changed = json.loads((verifier / CONFIG).read_text())
        changed['data']['events_per_channel'] = 6
        (verifier / CONFIG).write_text(json.dumps(changed))
    if scenario == 'tampered':
        metadata_path = scratch / 'prepared' / 'prepared.json'
        changed = json.loads(metadata_path.read_text())
        changed['feature_stats']['mean'][0] += 1
        metadata_path.write_text(json.dumps(changed))
        with tarfile.open(verifier / ARCHIVE, 'w:gz') as bundle:
            bundle.add(scratch / 'prepared', arcname='prepared')
    if scenario == 'synthetic':
        from src.prepare_pairwise import prepare
        # Explicit synthetic artifacts must not pass the real-data adoption gate.
        synthetic = tmp_path / 'synthetic'
        prepare(config, synthetic, synthetic=True)
        with tarfile.open(verifier / ARCHIVE, 'w:gz') as bundle:
            bundle.add(synthetic, arcname='prepared')
    before = (verifier / ARCHIVE).read_bytes()
    environment = os.environ.copy()
    environment['PATH'] = str(Path(sys.executable).parent) + os.pathsep + environment['PATH']
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    completed = subprocess.run(
        ['bash', 'run_prepare.sh', PAIR, CONFIG, ARCHIVE, ARCHIVE, RECEIPT, 'verify'],
        cwd=verifier, env=environment, capture_output=True, text=True, timeout=90)
    success = scenario in {'matching', 'explicit_revision'}
    assert (completed.returncode == 0) == success, completed.stdout + completed.stderr
    assert (verifier / ARCHIVE).read_bytes() == before  # success AND failure preserve the input
    status = json.loads((verifier / f'prepare_{PAIR}_status.json').read_text())
    receipt = json.loads((verifier / RECEIPT).read_text())
    assert status['success'] is receipt['success'] is success
    assert status['receipt_sha256'] == digest(verifier / RECEIPT)
    assert status['archive_sha256'] == receipt['archive_sha256'] == digest(verifier / ARCHIVE)
    assert receipt['provenance_kind'] == 'verified_existing_prepared'
    assert receipt['raw_archive'] is None
    assert not (verifier / 'input_paths.json').exists()
    assert not (verifier / 'selected_inputs').exists()
    assert not (verifier / 'raw').exists()
    if success:
        assert receipt['preprocessing'] == original_receipt['preprocessing']
        assert receipt['prepared_fingerprint'] == original_receipt['prepared_fingerprint']
        assert receipt['original_preparation_source_archive_sha256'] is None
        if scenario == 'explicit_revision':
            assert receipt['raw_archive_sha256'] is None
            assert 'does not establish' in receipt['revision_meaning']
        else:
            assert receipt['raw_archive_sha256'] == original_receipt['raw_archive_sha256']
    else:
        assert receipt['error']
