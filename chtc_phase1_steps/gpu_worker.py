#!/usr/bin/env python3
"""Verify one pair's prepared receipt, probe CUDA, then train or resume a run."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import traceback

try:
    from .runtime_utils import extract_safe_archive, metadata_fingerprint, require_basename, sha256_file, write_json
except ImportError:  # The deployment transfers these files into one directory.
    from runtime_utils import extract_safe_archive, metadata_fingerprint, require_basename, sha256_file, write_json


def cuda_probe():
    import numpy as np
    import torch
    import torch_geometric
    import torch_cluster
    from torch_geometric.data import Batch, Data
    from src.models.multispace import MultiSpaceEncoder
    from src.losses.multitask import MultiTaskObjective
    versions = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                'torch_cuda_build': torch.version.cuda, 'pyg': torch_geometric.__version__,
                'torch_cluster': torch_cluster.__version__, 'cuda_available': torch.cuda.is_available()}
    print('Runtime:', json.dumps(versions, indent=2), flush=True)
    if not versions['cuda_available']:
        raise RuntimeError('This is a GPU job, but CUDA is unavailable. No CPU fallback is allowed.')
    versions['gpu'] = torch.cuda.get_device_name(0)
    print('GPU:', versions['gpu'], flush=True)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    # Small synthetic graphs only check compiled CUDA/PyG operators and the real
    # model/objective API. They are not training data or physics-performance evidence.
    def graphs():
        return Batch.from_data_list([
            Data(x=torch.randn(n, 3), energy=torch.rand(n) + .1, summary=torch.randn(1, 2))
            for n in (9, 12)
        ]).to(device)
    model = MultiSpaceEncoder().to(device)
    objective = MultiTaskObjective().to(device)
    out1, out2 = model(graphs()), model(graphs())
    targets = {'energy': torch.zeros(2, 2, device=device),
               'eta': torch.full((2, 8), 1/8, device=device),
               'phi': torch.zeros(2, 4, device=device),
               'local': torch.zeros(2, 4, device=device)}
    loss = objective(out1, out2, targets, targets)['loss']
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                         for p in objective.parameters()):
        raise RuntimeError('CUDA model/objective probe failed.')
    versions['cuda_forward_backward_probe'] = 'passed'
    print('CUDA_FORWARD_BACKWARD_OK', flush=True)
    del model, objective, out1, out2, targets, loss
    gc.collect()
    torch.cuda.empty_cache()
    return versions


def resolved_config(requested, metadata):
    """Resolve only preparation-controlled fields, exactly as the trainer does.

    Explicit values must agree with frozen artifacts. None permits a prepared
    value only in data/grid/targets, never optimizer or augmentation settings.
    """
    effective = copy.deepcopy(requested)
    preparation = metadata['preparation_config']
    for section in ('data', 'grid', 'targets'):
        for key, value in requested[section].items():
            if value is not None and value != preparation[section][key]:
                raise ValueError(f'{section}.{key} disagrees with the prepared pair.')
        effective[section] = copy.deepcopy(preparation[section])
    effective['prepared_dir'] = str(Path('prepared').resolve())
    return effective


def verify_inputs(args, detail):
    """Complete data/config/resume checks before any CUDA probe or training."""
    from src.config import load_config
    from src.data.events import load_prepared

    detail['stage'] = 'verify_receipt'
    if Path('hep_ssl-code.tar.gz').is_file():
        detail['code_archive_sha256'] = sha256_file('hep_ssl-code.tar.gz')
    config = load_config(args.config)
    pair_id = '_'.join(config['data']['channels'])
    detail.update(pair_id=pair_id, configured_total_epochs=config['training']['epochs'])
    if pair_id not in {'ggf_ttbar', 'ggf_dihiggs', 'ttbar_dihiggs'}:
        raise ValueError('Run configuration does not identify one of the three declared pairs.')
    if config['training']['device'] != 'cuda':
        raise ValueError('GPU deployment config must explicitly require cuda.')
    if args.stop_after_epoch < 0 or args.stop_after_epoch > config['training']['epochs']:
        raise ValueError('stop_after_epoch must be 0 (full run) or within the configured horizon.')
    receipt = json.loads(Path(args.receipt).read_text())
    if receipt.get('success') is not True or receipt.get('synthetic') is not False:
        raise ValueError('A successful real-data preparation receipt is required.')
    if receipt.get('pair_id') != pair_id:
        raise ValueError('Preparation receipt pair_id differs from this run configuration.')
    detail['preparation_code_archive_sha256'] = receipt.get('code_archive_sha256')
    if sha256_file(args.prepared_archive) != receipt.get('archive_sha256'):
        raise ValueError('Prepared archive SHA-256 does not match its receipt.')
    detail['prepared_archive_sha256'] = receipt['archive_sha256']
    if Path('prepared').exists():
        raise FileExistsError('The scratch prepared directory already exists; refuse to mix artifacts.')
    detail['stage'] = 'extract_prepared'
    extract_safe_archive(args.prepared_archive, '.', expected_root='prepared')
    detail['stage'] = 'verify_prepared'
    prepared = load_prepared('prepared')
    metadata = prepared.metadata
    fingerprint = metadata_fingerprint(metadata)
    if metadata.get('synthetic') is not False:
        raise ValueError('Synthetic prepared inputs cannot be used for this GPU deployment.')
    if str(metadata['data']['dataset_revision']).lower().startswith('synthetic'):
        raise ValueError('A synthetic data revision cannot be used for this GPU deployment.')
    if metadata['data']['channels'] != config['data']['channels']:
        raise ValueError('Prepared channel order differs from the requested label convention.')
    if fingerprint != receipt.get('prepared_fingerprint'):
        raise ValueError('Loaded prepared fingerprint differs from the receipt.')
    if receipt.get('preprocessing') != metadata:
        raise ValueError('Receipt preprocessing metadata differs from the loaded prepared artifact.')
    effective = resolved_config(config, metadata)
    detail.update(prepared_fingerprint=fingerprint, synthetic=False,
                  effective_config=effective, requested_config=config)
    resume_path = None
    if args.resume_archive != '-':
        from src.training.checkpoint import assert_prepared_matches, load_checkpoint

        detail['stage'] = 'verify_resume'
        run_dir = Path('outputs') / args.run_id
        if run_dir.exists():
            raise FileExistsError('Resume output directory already exists; refuse to merge two runs.')
        extract_safe_archive(args.resume_archive, 'outputs', expected_root=args.run_id)
        resume_path = run_dir / 'checkpoints' / 'last.pt'
        prior = load_checkpoint(resume_path)
        assert_prepared_matches(prior, metadata)
        comparable = copy.deepcopy(effective)
        comparable['prepared_dir'] = prior['config']['prepared_dir']
        if comparable != prior['config']:
            raise ValueError('Resume configuration differs from the requested run; augmentation, model, '
                             'loss, seed, optimizer, and total scheduler horizon must match.')
        disk_config = json.loads((run_dir / 'config.json').read_text())
        if disk_config != prior['config']:
            raise ValueError('Archived run config differs from its checkpoint.')
        completed = prior['epoch'] + 1
        target_epoch = args.stop_after_epoch or config['training']['epochs']
        if completed >= target_epoch:
            raise ValueError('Resume checkpoint has already reached the requested stopping epoch.')
        detail.update(completed_epochs=completed, resumed_from_epoch=completed,
                      previous_source_sha256=prior.get('source_sha256'))
    return config, prepared, resume_path


def checkpoint_details(state):
    last = state['history'][-1]
    return {'completed_epochs': state['epoch'] + 1,
            'configured_total_epochs': state['config']['training']['epochs'],
            'global_step': state['global_step'], 'train_loss': last['train']['loss'],
            'validation_loss': last['val']['loss'],
            'prepared_fingerprint': state['prepared_fingerprint'],
            'source_sha256': state.get('source_sha256'),
            'lambda_ranges': {name: [min(values), max(values)]
                              for name, values in last.get('lambda', {}).items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True, type=require_basename)
    ap.add_argument('--config', required=True, type=require_basename)
    ap.add_argument('--prepared-archive', required=True, type=require_basename)
    ap.add_argument('--receipt', required=True, type=require_basename)
    ap.add_argument('--stop-after-epoch', type=int, default=0)
    ap.add_argument('--resume-archive', default='-')
    args = ap.parse_args(argv)
    if args.resume_archive != '-':
        require_basename(args.resume_archive)
    run_dir = Path('outputs') / args.run_id
    began = time.monotonic()
    detail = {'run_id': args.run_id, 'config_basename': args.config,
              'prepared_archive': args.prepared_archive, 'receipt_basename': args.receipt,
              'resume_archive': args.resume_archive, 'stop_after_epoch': args.stop_after_epoch,
              'started_at': datetime.now(timezone.utc).isoformat(), 'completed_epochs': 0,
              'success': False, 'stage': 'initialization'}
    exit_code = 1
    try:
        config, prepared, resume_path = verify_inputs(args, detail)
        detail['stage'] = 'cuda_probe'
        detail['runtime'] = cuda_probe()
        from src.training.trainer import train
        from src.training.checkpoint import assert_prepared_matches, load_checkpoint, source_state

        detail['current_source'] = source_state()
        detail['stage'] = 'training'
        path = train(None if resume_path else config, 'prepared', run_dir,
                     resume=resume_path, stop_after_epoch=args.stop_after_epoch or None)
        state = load_checkpoint(path)
        assert_prepared_matches(state, prepared.metadata)
        detail.update(checkpoint_details(state))
        detail.update(success=True, stage='complete',
                      training_complete=detail['completed_epochs'] == detail['configured_total_epochs'])
        exit_code = 0
    except Exception as exc:
        detail.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(detail['traceback'], file=sys.stderr, flush=True)
        checkpoint = run_dir / 'checkpoints' / 'last.pt'
        if detail['stage'] == 'training' and checkpoint.is_file():
            try:
                from src.training.checkpoint import load_checkpoint
                detail.update(checkpoint_details(load_checkpoint(checkpoint)))
            except Exception as checkpoint_error:
                detail['checkpoint_inspection_error'] = str(checkpoint_error)
    finally:
        detail.update(exit_code=exit_code, elapsed_seconds=time.monotonic() - began,
                      finished_at=datetime.now(timezone.utc).isoformat())
        # Fresh train() owns directory creation. Only after it returns or fails
        # may the worker create a diagnostics-only output directory.
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / 'job_details.json', detail)
    print('JOB_COMPLETE:', json.dumps(detail, indent=2, allow_nan=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
