#!/usr/bin/env python3
"""CUDA environment probe, then the existing trainer: first epoch or resume."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=('first', 'continue'))
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    versions = cuda_probe()
    from src.config import load_config
    from src.training.trainer import train
    from src.training.checkpoint import load_checkpoint
    run_dir = Path('outputs') / args.run_id
    began = time.monotonic()
    if args.stage == 'first':
        config = load_config('pairwise_chtc.json')
        if config['training']['device'] != 'cuda':
            raise ValueError('Deployment config must require cuda.')
        # Same data, model, batch size, and total epoch count as the formal run.
        path = train(config, 'prepared', run_dir, stop_after_epoch=1)
    else:
        checkpoint = run_dir / 'checkpoints' / 'last.pt'
        prior = load_checkpoint(checkpoint)
        if prior['config']['training']['device'] != 'cuda':
            raise ValueError('Resume checkpoint is not configured for CUDA.')
        if prior['preprocessing'].get('synthetic'):
            raise ValueError('Do not use synthetic prepared data for this deployment.')
        del prior
        path = train(None, 'prepared', run_dir, resume=checkpoint)
    elapsed = time.monotonic() - began
    state = load_checkpoint(path)
    if state['preprocessing'].get('synthetic'):
        raise ValueError('The produced run used synthetic data, not the requested real cache.')
    last = state['history'][-1]
    detail = {'runtime': versions, 'completed_epochs': state['epoch'] + 1,
              'configured_total_epochs': state['config']['training']['epochs'],
              'global_step': state['global_step'], 'train_loss': last['train']['loss'],
              'validation_loss': last['val']['loss'], 'elapsed_seconds': elapsed,
              'prepared_fingerprint': state['prepared_fingerprint'],
              'source_sha256': state.get('source_sha256'),
              'lambda_ranges': {name: [min(values), max(values)] for name, values in last['lambda'].items()}}
    (run_dir / (args.stage + '_details.json')).write_text(json.dumps(detail, indent=2, allow_nan=False) + '\n')
    print('STAGE_COMPLETE:', json.dumps(detail, indent=2), flush=True)


if __name__ == '__main__':
    main()
