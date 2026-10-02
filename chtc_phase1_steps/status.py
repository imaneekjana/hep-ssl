#!/usr/bin/env python3
"""Read small returned files on the access point. Does not run training there."""
import argparse
import json
from pathlib import Path
import sys
import tarfile


def check(stage):
    status_path = Path(stage + '_status.json')
    if not status_path.is_file():
        raise RuntimeError(f'{status_path} has not returned. Check condor_q and the job .log/.err.')
    s = json.loads(status_path.read_text())
    if s.get('success') is not True or s.get('exit_code') != 0:
        raise RuntimeError(f'{stage} did not succeed: {json.dumps(s)}. Read its .err and .out.')
    if stage == 'prepare':
        d = json.loads(Path('prepare_details.json').read_text())
        if d.get('success') is not True or d.get('synthetic') is not False:
            raise RuntimeError('Preparation details do not confirm successful real-data preparation.')
        print('PREPARE_OK')
        for k in ('events', 'excluded_events', 'grid_shape', 'eta_range', 'range_source',
                  'dataset_revision', 'revision_meaning', 'prepared_archive_bytes'):
            print(f'{k}: {d.get(k)}')
        for row in d['counts']:
            print(row)
        size = d['prepared_archive_bytes']
        if size > 30_000_000_000:
            raise RuntimeError('Prepared archive exceeds the OSDF 30 GB guide range; change staged transfers to file:/// before GPU submission.')
        if size < 1_000_000_000:
            print('NOTE: prepared archive is under 1 GB. It may be moved to /home and referenced locally; /staging is intended for large inputs.')
        return
    archive = Path(s['archive'])
    with tarfile.open(archive, 'r:gz') as tf:
        name = s['run_id'] + '/'
        member_names = set(tf.getnames())
        if name + 'checkpoints/last.pt' not in member_names or name + 'checkpoints/best.pt' not in member_names:
            raise RuntimeError('Result archive lacks required checkpoints.')
        history = json.load(tf.extractfile(name + 'history.json'))
        cfg = json.load(tf.extractfile(name + 'config.json'))
        last = history[-1]
    if cfg['training']['device'] != 'cuda':
        raise RuntimeError('Run did not require CUDA.')
    completed = last['epoch'] + 1
    if stage == 'first' and completed != 1:
        raise RuntimeError('First stage did not finish exactly one epoch.')
    if stage == 'continue' and completed != cfg['training']['epochs']:
        raise RuntimeError('Continuation did not complete all configured epochs.')
    print('FIRST_EPOCH_OK' if stage == 'first' else 'TRAINING_COMPLETE')
    print(f'completed epochs: {completed}/{cfg["training"]["epochs"]}')
    print('training total loss:', last['train']['loss'])
    print('validation total loss:', last['val']['loss'])
    for name, values in last['lambda'].items():
        print(f'Lambda {name}: min={min(values):.6g}, max={max(values):.6g}, trace={sum(values):.6g}')
    print('runtime:', s.get('details', {}).get('runtime', {}))
    print('elapsed seconds:', s.get('details', {}).get('elapsed_seconds'))
    if stage == 'first':
        elapsed = s.get('details', {}).get('elapsed_seconds')
        if elapsed:
            remaining_hours = (cfg['training']['epochs'] - 1) * elapsed / 3600
            print(f'Rough remaining-time estimate using first-stage duration: {remaining_hours:.2f} hours (not a guarantee).')
            if remaining_hours > 20:
                print('ACTION: Estimated continuation approaches/exceeds 24h. Set +GPUJobLength="long" in 03_continue.sub before proceeding.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'first', 'continue'))
    args = ap.parse_args()
    try:
        check(args.stage)
    except Exception as exc:
        print('NOT_READY:', exc, file=sys.stderr)
        raise SystemExit(1)
