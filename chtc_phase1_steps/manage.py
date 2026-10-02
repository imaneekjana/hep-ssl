#!/usr/bin/env python3
"""Manage one generated deployment from its /home submit directory."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

try:
    from .deployment_checks import (check_prepared, completed_job, local_path, read_json,
                                    validate_record, verify_deployment)
    from .runtime_utils import sha256_file, write_json
except ImportError:
    from deployment_checks import (check_prepared, completed_job, local_path, read_json,
                                   validate_record, verify_deployment)
    from runtime_utils import sha256_file, write_json


def attempt_directory(root, stage):
    token = stage + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
    path = root / 'attempts' / token
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_rows(path, rows):
    for row in rows:
        if any(not value or re.search(r'[\s,;"\x00]', str(value)) for value in row):
            raise ValueError('Condor manifest values must not contain whitespace/delimiters')
    path.write_text(''.join('\t'.join(map(str, row)) + '\n' for row in rows))


def require_submit_directory(root, info):
    expected = Path(info['remote_directory'])
    if root.resolve() != expected:
        raise ValueError(f'Run submission from {expected}; current directory is {root}')


def submit(root, info, attempt, template, manifest, count, *, dry_run=False, macros=()):
    relative = str(attempt.relative_to(root))
    command = ['condor_submit', '-terse', template]
    # Bare CLI assignments precede the submit file and can be overwritten by its
    # defaults. -append places these overrides immediately before queue.
    for value in ['manifest=' + str(manifest.relative_to(root)), 'attempt=' + relative, *macros]:
        command.extend(['-append', value])
    print(shlex.join(command))
    if dry_run:
        print('DRY_RUN: wrote a local plan; no submission. Preparation checks may cache completion records.')
        return None
    require_submit_directory(root, info)
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    (attempt / 'submit.stdout').write_text(result.stdout)
    (attempt / 'submit.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'condor_submit failed; inspect {relative}/submit.stderr')
    match = re.fullmatch(r'\s*(\d+)\.(\d+)(?:\s*-\s*(\d+)\.(\d+))?\s*', result.stdout)
    if not match:
        raise RuntimeError(f'Unexpected condor_submit response; inspect {relative}/submit.stdout and condor_q before retrying')
    cluster, first = int(match[1]), int(match[2])
    last_cluster, last = (int(match[3]), int(match[4])) if match[3] else (cluster, first)
    if last_cluster != cluster or last - first + 1 != count:
        raise RuntimeError('Unexpected submitted job count; inspect condor_q before retrying')
    jobs = [f'{cluster}.{proc}' for proc in range(first, last + 1)]
    print(f'SUBMITTED {count} jobs: {jobs[0]} .. {jobs[-1]}')
    return jobs


def prepare_jobs(root, *, pair=None, retry=False, dry_run=False):
    info, _ = verify_deployment(root)
    registry = read_json(root / 'prepared_registry.json')
    if pair and pair not in info['pairs']:
        raise ValueError('Unknown pair')
    if retry and pair is None:
        raise ValueError('--retry requires exactly one --pair')
    selected = []
    for pair_id in ([pair] if pair else info['pairs']):
        if pair_id in registry:
            record = registry[pair_id]
            if retry:
                # A removed job (status 3) also needs confirmation it is no longer queued.
                ad = completed_job(record['job_id'], allow_removed=True)
                if ad.get('JobStatus') == 4 and ad.get('ExitCode') == 0 and not ad.get('ExitBySignal', False):
                    raise ValueError('Successful prepared data is immutable; reuse it or generate a new deployment')
            else:
                config = read_json(root / info['pairs'][pair_id]['config_path'])
                validate_record(root, pair_id, record, config)
                print(f'REUSE {pair_id}: {record["prepared_url"]}')
                continue
        selected.append(pair_id)
    if not selected:
        print('All prepared tasks already succeeded; no CPU jobs submitted.')
        return
    if not dry_run:
        require_submit_directory(root, info)
    attempt = attempt_directory(root, 'prepare')
    rows, records = [], {}
    for pair_id in selected:
        pair_config = info['pairs'][pair_id]
        name = f'prepared_{pair_id}_{attempt.name}.tar.gz'
        url = ('osdf:///chtc' if info['settings']['transfer_protocol'] == 'osdf' else 'file://') + info['stage_directory'] + '/' + name
        receipt = f'prepare_{pair_id}_details.json'
        config_path = pair_config['config_path']
        rows.append((pair_id, config_path, Path(config_path).name, name, url, receipt))
        records[pair_id] = {'prepared_name': name, 'prepared_url': url,
                            'status_path': str((attempt / f'prepare_{pair_id}_status.json').relative_to(root)),
                            'receipt_path': str((attempt / receipt).relative_to(root)),
                            'config_sha256': pair_config['config_sha256'], 'source_sha256': info['source_sha256'],
                            'input_paths_sha256': sha256_file(root / 'input_paths.json')}
    manifest = attempt / 'prepare.tsv'
    write_rows(manifest, rows)
    write_json(attempt / 'plan.json', {'stage': 'prepare', 'pairs': records})
    jobs = submit(root, info, attempt, '01_prepare.sub', manifest, len(rows), dry_run=dry_run)
    if jobs:
        for pair_id, job_id in zip(selected, jobs):
            records[pair_id]['job_id'] = job_id
        write_json(attempt / 'submission.json', {'stage': 'prepare', 'pairs': records})
        registry.update(records)
        write_json(root / 'prepared_registry.json', registry)


def prior_training(root):
    latest = {}
    for path in sorted((root / 'attempts').glob('train_*/submission.json')):
        submission = read_json(path)
        for row in submission['runs']:
            latest[row['run_id']] = {**row, 'attempt': path.parent}
    return latest


def reuse_archive(root, *, pair, archive, dry_run=False):
    """Validate an existing prepared archive on a CPU node, without regenerating it."""
    info, _ = verify_deployment(root)
    registry = read_json(root / 'prepared_registry.json')
    if pair not in info['pairs'] or pair in registry:
        raise ValueError('Choose an unregistered pair in a new deployment')
    if not re.fullmatch(r'(?:osdf:///chtc|file://)/staging/[A-Za-z0-9_./-]+', archive):
        raise ValueError('--archive must be an osdf:///chtc/staging/... or file:///staging/... URL')
    if '..' in archive.split('/'):
        raise ValueError('Parent traversal is forbidden')
    if not dry_run:
        require_submit_directory(root, info)
    attempt = attempt_directory(root, 'verify')
    name = archive.rsplit('/', 1)[-1]
    config = info['pairs'][pair]
    receipt = f'prepare_{pair}_details.json'
    path = attempt / 'verify.tsv'
    write_rows(path, [(pair, config['config_path'], Path(config['config_path']).name, name, archive, receipt)])
    record = {'prepared_name': name, 'prepared_url': archive,
              'status_path': str((attempt / f'prepare_{pair}_status.json').relative_to(root)),
              'receipt_path': str((attempt / receipt).relative_to(root)),
              'config_sha256': config['config_sha256'], 'source_sha256': info['source_sha256'],
              'input_paths_sha256': None, 'operation': 'verify'}
    write_json(attempt / 'plan.json', {'stage': 'verify', 'pairs': {pair: record}})
    jobs = submit(root, info, attempt, '01_verify.sub', path, 1, dry_run=dry_run)
    if jobs:
        record['job_id'] = jobs[0]
        registry[pair] = record
        write_json(attempt / 'submission.json', {'stage': 'verify', 'pairs': {pair: record}})
        write_json(root / 'prepared_registry.json', registry)


def train_jobs(root, *, run_id=None, retry=False, remaining=False, stop_after_epoch=0,
               resume_from=None, dry_run=False):
    info, manifest = verify_deployment(root)
    if run_id and run_id not in {r['run_id'] for r in manifest}:
        raise ValueError('Unknown run_id')
    if remaining and run_id:
        raise ValueError('--remaining and --run-id are mutually exclusive')
    if (retry or stop_after_epoch or resume_from) and not run_id:
        raise ValueError('Retry, short run and resume each require exactly one --run-id')
    if stop_after_epoch < 0 or stop_after_epoch >= 18:
        raise ValueError('--stop-after-epoch must be 1..17, or omitted for all 18 epochs')
    # Check all three tasks even for a selected smoke run. Never dispatch early.
    registry = check_prepared(root)
    prior = prior_training(root)
    selected = [r for r in manifest if (not run_id or r['run_id'] == run_id)
                and (not remaining or r['run_id'] not in prior)]
    for row in selected:
        old = prior.get(row['run_id'])
        if old:
            if not retry:
                raise ValueError(f'{row["run_id"]} was already submitted; use --remaining or a selected --retry')
            completed_job(old['job_id'], allow_removed=True)  # Never duplicate a running/held job.
    if not selected:
        print('No remaining runs to submit.')
        return
    # Every row binds its own config to its pair's shared immutable receipt/archive.
    if not dry_run:
        require_submit_directory(root, info)
    attempt = attempt_directory(root, 'train')
    macros = [f'stop_after_epoch={stop_after_epoch}']
    if resume_from:
        source = Path(resume_from).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = attempt / 'resume_input.tar.gz'
        shutil.copy2(source, target)
        macros.extend(['resume_basename=resume_input.tar.gz', 'resume_transfer=, ' + str(target.relative_to(root))])
    rows = []
    for row in selected:
        record = registry[row['prepared_id']]
        rows.append((row['run_id'], row['config_path'], row['config_basename'], record['prepared_name'],
                     record['prepared_url'], record['receipt_path'], Path(record['receipt_path']).name))
    path = attempt / 'train.tsv'
    write_rows(path, rows)
    plan = {'stage': 'train', 'stop_after_epoch': stop_after_epoch,
            'runs': [{'run_id': r['run_id'], 'prepared_id': r['prepared_id'],
                      'prepared_fingerprint': registry[r['prepared_id']]['metadata_fingerprint']} for r in selected]}
    write_json(attempt / 'plan.json', plan)
    jobs = submit(root, info, attempt, '02_train.sub', path, len(rows), dry_run=dry_run, macros=macros)
    if jobs:
        for row, job in zip(plan['runs'], jobs):
            row['job_id'] = job
        write_json(attempt / 'submission.json', plan)


def show_status(root, run_id=None):
    info, manifest = verify_deployment(root)
    registry = read_json(root / 'prepared_registry.json')
    for pair in info['pairs']:
        record = registry.get(pair)
        state = 'NOT_SUBMITTED'
        if record:
            path = root / record['status_path']
            state = 'PENDING' if not path.exists() else ('RETURNED_OK' if read_json(path).get('success') else 'FAILED')
        print(f'prepare {pair}: {state}')
    previous = prior_training(root)
    if run_id and run_id not in {r['run_id'] for r in manifest}:
        raise ValueError('Unknown run_id')
    for row in manifest:
        name = row['run_id']
        if run_id and name != run_id:
            continue
        state, detail = 'NOT_SUBMITTED', ''
        if name in previous:
            old = previous[name]
            path = old['attempt'] / row['status_file']
            state = 'PENDING'
            detail = f'job={old["job_id"]} directory={old["attempt"].relative_to(root)}'
            if path.is_file():
                status = read_json(path)
                state = 'FAILED'
                if status.get('success') and status.get('exit_code') == 0:
                    state = 'COMPLETE' if status.get('completed_epochs') == 18 else 'SHORT_RUN_COMPLETE'
                if run_id:
                    print(json.dumps(status, indent=2))
        print(f'{name}: {state} {detail}')


def locations(root):
    info = read_json(root / 'deployment.json')
    settings = info['settings']
    command = ' && '.join(['test -r ' + shlex.quote(settings['raw_archive']),
                           'test -r ' + shlex.quote(settings['container_image']),
                           'mkdir -p ' + shlex.quote(info['stage_directory'])])
    print('Run this yourself before preparation (no SSH is executed by this script):')
    print(shlex.join(['ssh', settings['transfer_login'], command]))
    print('Submission directory:', info['remote_directory'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--pair')
    prep.add_argument('--retry', action='store_true')
    prep.add_argument('--dry-run', action='store_true')
    subs.add_parser('check-prepared')
    subs.add_parser('locations')
    reuse = subs.add_parser('reuse')
    reuse.add_argument('--pair', required=True)
    reuse.add_argument('--archive', required=True, help='Existing prepared staging URL; never a raw-data archive')
    reuse.add_argument('--dry-run', action='store_true')
    train = subs.add_parser('train')
    train.add_argument('--run-id')
    train.add_argument('--retry', action='store_true')
    train.add_argument('--remaining', action='store_true')
    train.add_argument('--stop-after-epoch', type=int, default=0)
    train.add_argument('--resume-from')
    train.add_argument('--dry-run', action='store_true')
    status = subs.add_parser('status')
    status.add_argument('--run-id')
    args = vars(parser.parse_args())
    command = args.pop('command')
    root = Path.cwd().resolve()
    if command == 'prepare':
        prepare_jobs(root, **args)
    elif command == 'train':
        train_jobs(root, **args)
    elif command == 'reuse':
        reuse_archive(root, **args)
    elif command == 'check-prepared':
        registry = check_prepared(root)
        for pair, record in registry.items():
            print(f'PREPARE_OK {pair} {record["metadata_fingerprint"]} {record["prepared_url"]}')
    elif command == 'status':
        show_status(root, **args)
    else:
        locations(root)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print('NOT_READY:', exc, file=sys.stderr)
        raise SystemExit(1)
