#!/usr/bin/env python3
"""Prepare one real channel pair with existing physics, then seal its receipt."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import tarfile
import traceback
try:
    from .runtime_utils import extract_safe_archive, metadata_fingerprint, require_basename, sha256_file, write_json
except ImportError:
    from runtime_utils import extract_safe_archive, metadata_fingerprint, require_basename, sha256_file, write_json

SUPPORTED_CHANNELS = {'ggf', 'ttbar', 'dihiggs'}


def discover_files(root, channel, pileup, override=None):
    root = Path(root).resolve()
    token = f'{channel}_{pileup}_calo_hits'
    if override:
        relative = Path(override)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('input_paths.json paths must be relative to raw/ without parent traversal.')
        loc = (root / relative).resolve()
        if not loc.is_relative_to(root):
            raise ValueError('input_paths.json paths must stay within the extracted raw/ directory.')
        files = list(loc.rglob('*.parquet')) if loc.is_dir() else [loc]
        files = [p for p in files if p.is_file() and p.suffix == '.parquet']
    else:
        files = [p for p in root.rglob('*.parquet')
                 if token in p.relative_to(root).parts
                 and re.match(r'^train(?:[-_.]|$)', p.stem) and p.is_file()]
    files = sorted(set(p.resolve() for p in files))
    if not files:
        examples = [str(p.relative_to(root)) for p in root.rglob('*.parquet')][:30]
        raise ValueError(f'No verified train calo-hit files for {token}. Actual Parquet examples: {examples}. '
                         'Set input_paths.json to the exact per-channel directory shown in this log.')
    parents = {p.parent for p in files}
    if not override and len(parents) != 1:
        raise ValueError(f'Multiple physical shard directories match {token}: {sorted(map(str, parents))}. '
                         'Select one explicitly in input_paths.json.')
    if any(not p.is_relative_to(root) for p in files):
        raise ValueError('A dataset symlink resolves outside raw/. Supply a self-contained archive.')
    return files


def _compatibility(config):
    return {section: copy.deepcopy(config[section]) for section in ('data', 'grid', 'targets')}


def _prepared_details(prepared, details):
    metadata = prepared.metadata
    if metadata.get('synthetic') is not False:
        raise RuntimeError('Preparation metadata must explicitly identify non-synthetic input.')
    counts = Counter((row['channel'], row['split'], row['usable']) for row in metadata['manifest'])
    details.update(
        synthetic=False, events=len(prepared.events), excluded_events=metadata['excluded_event_count'],
        manifest_hash=metadata['manifest_hash'], metadata_hash=metadata['metadata_hash'],
        metadata_fingerprint=metadata_fingerprint(metadata),
        prepared_fingerprint=metadata_fingerprint(metadata), preprocessing=metadata,
        preparation_config=metadata['preparation_config'], compatibility=_compatibility(metadata['preparation_config']),
        grid_shape=list(prepared.grid.shape), eta_range=[prepared.grid.eta_edges[0], prepared.grid.eta_edges[-1]],
        range_source=prepared.grid.range_source,
        counts=[{'channel': key[0], 'split': key[1], 'usable': key[2], 'count': count}
                for key, count in sorted(counts.items())],
        prepared_uncompressed_bytes=sum(path.stat().st_size for path in Path('prepared').rglob('*') if path.is_file()))


def verify_existing(args, details):
    """Validate a frozen legacy/current archive without fitting or re-archiving."""
    from src.config import load_config
    from src.data.events import load_prepared
    cfg = load_config(args.config)
    if args.pair_id != '_'.join(cfg['data']['channels']):
        raise ValueError('PAIR_ID must match configured channels in label order.')
    if args.archive != args.prepared_archive:
        raise ValueError('Verification must preserve the existing archive basename and bytes.')
    details['requested_compatibility'] = _compatibility(cfg)
    extract_safe_archive(args.archive, '.', expected_root='prepared')
    prepared = load_prepared('prepared')
    metadata = prepared.metadata
    if metadata.get('synthetic') is not False:
        raise ValueError('Synthetic prepared archives cannot be adopted for real-data runs.')
    preparation = metadata['preparation_config']
    for section in ('data', 'grid', 'targets'):
        for name, value in cfg[section].items():
            if value is not None and value != preparation[section].get(name):
                raise ValueError(f'Incompatible prepared configuration: {section}.{name}')
    if preparation['data']['channels'] != args.pair_id.split('_'):
        raise ValueError('Prepared metadata belongs to a different channel pair.')
    revision = metadata['data']['dataset_revision']
    match = re.fullmatch(r'local-cache-sha256:([0-9a-fA-F]{64})', str(revision))
    raw_digest = match.group(1).lower() if match else None
    meaning = ('raw-archive SHA256 recovered from frozen local-cache revision; original raw bytes were not re-read'
               if match else 'frozen explicit dataset revision; metadata does not establish the original raw-archive SHA256')
    details.update(dataset_revision=revision, raw_archive=None, raw_archive_sha256=raw_digest,
                   revision_meaning=meaning, requested_channels=cfg['data']['channels'],
                   requested_events_per_channel=cfg['data']['events_per_channel'],
                   verification_source_archive_sha256=details['code_archive_sha256'],
                   original_preparation_source_archive_sha256=None)
    _prepared_details(prepared, details)
    size = Path(args.archive).stat().st_size
    details.update(success=True, archive_sha256=sha256_file(args.archive), archive_bytes=size, prepared_archive_bytes=size)
    write_json(args.receipt, details)
    print(json.dumps({'pair_id': args.pair_id, 'verified_existing': True, 'archive': args.archive,
                      'archive_sha256': details['archive_sha256'], 'archive_bytes': size,
                      'metadata_fingerprint': details['metadata_fingerprint']}, indent=2), flush=True)


def prepare_and_archive(args, details):
    from src.config import load_config
    from src.prepare_pairwise import prepare
    import polars as pl

    cfg = load_config(args.config)
    if args.pair_id != '_'.join(cfg['data']['channels']):
        raise ValueError('PAIR_ID must equal the two configured channels joined by an underscore in label order.')
    if not set(cfg['data']['channels']) <= SUPPORTED_CHANNELS:
        raise ValueError('Unsupported channels in preparation configuration.')
    details['requested_compatibility'] = _compatibility(cfg)
    overrides = json.loads(Path(args.input_paths).read_text())
    if (not isinstance(overrides, dict) or set(overrides) - SUPPORTED_CHANNELS
            or any(not isinstance(value, str) or not value for value in overrides.values())):
        raise ValueError('input_paths.json must map supported channel names to nonempty raw-relative paths; all three channel overrides are allowed.')
    archive_hash = sha256_file(args.archive)
    supplied_revision = cfg['data'].get('dataset_revision')
    if supplied_revision is None:
        cfg['data']['dataset_revision'] = 'local-cache-sha256:' + archive_hash
    if str(cfg['data']['dataset_revision']).startswith('synthetic'):
        raise ValueError('Synthetic dataset revisions are not allowed in real preparation jobs.')
    revision_note = ('user-supplied upstream/local revision, retained unchanged' if supplied_revision
                     else 'exact local raw-archive SHA256; NOT a claim about upstream Hugging Face revision')
    details.update(raw_archive_sha256=archive_hash, dataset_revision=cfg['data']['dataset_revision'],
                   revision_meaning=revision_note, requested_channels=cfg['data']['channels'],
                   requested_events_per_channel=cfg['data']['events_per_channel'])
    print('Dataset snapshot:', cfg['data']['dataset_revision'], flush=True)
    print('Snapshot meaning:', revision_note, flush=True)
    source_records, inputs = {}, {}
    required = {'event_id', 'x', 'y', 'z', 'total_energy'}
    raw_root = Path(args.raw_root).resolve()
    for channel in cfg['data']['channels']:
        files = discover_files(raw_root, channel, cfg['data']['pileup'], overrides.get(channel))
        folder = Path('selected_inputs') / channel
        folder.mkdir(parents=True, exist_ok=False)
        selected, remaining = [], cfg['data']['events_per_channel']
        for i, path in enumerate(files):
            schema = pl.read_parquet_schema(path)
            if not required.issubset(schema):
                raise ValueError(f'{path} lacks {sorted(required - set(schema))}. No IDs will be fabricated.')
            for field in ('x', 'y', 'z', 'total_energy'):
                dtype = schema[field]
                if not isinstance(dtype, (pl.List, pl.Array)) or not dtype.inner.is_numeric():
                    raise ValueError(f'{path}: {field} must be a numeric per-event list/array, got {dtype}.')
            if isinstance(schema['event_id'], (pl.List, pl.Array, pl.Struct)):
                raise ValueError(f'{path}: event_id must be a scalar upstream identity.')
            # Read only the IDs needed to bound selected shards. Core prepare
            # independently reads the same bounded queue and validates identities.
            rows = pl.read_parquet(path, columns=['event_id'], n_rows=remaining).height
            if rows:
                (folder / f'{i:06d}_{path.name}').symlink_to(path)
                selected.append(str(path.relative_to(raw_root)))
                remaining -= rows
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f'{channel}: only {cfg["data"]["events_per_channel"] - remaining} local events are available; requested {cfg["data"]["events_per_channel"]}.')
        inputs[channel] = str(folder.resolve())
        source_records[channel] = selected
        details['selected_source_files'] = source_records
        write_json(args.receipt, details)
        print(f'{channel}: {len(selected)} verified train shards selected', flush=True)
        for item in selected[:5]:
            print(' ', item, flush=True)
    prepared = prepare(cfg, 'prepared', inputs=inputs, synthetic=False)
    _prepared_details(prepared, details)
    # Success is never published before tar closure, digest and atomic rename.
    archive = Path(args.prepared_archive)
    temporary = archive.with_name(archive.name + '.partial')
    try:
        with tarfile.open(temporary, 'w:gz') as bundle:
            bundle.add('prepared', arcname='prepared')
        digest = sha256_file(temporary)
        size = temporary.stat().st_size
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    details.update(success=True, archive_sha256=digest, archive_bytes=size, prepared_archive_bytes=size)
    write_json(args.receipt, details)
    print(json.dumps({key: value for key, value in details.items()
                      if key not in {'preprocessing', 'preparation_config', 'selected_source_files'}}, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-id', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--input-paths', required=True)
    parser.add_argument('--prepared-archive', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--code-archive', default='hep_ssl-code.tar.gz')
    parser.add_argument('--verify-existing', action='store_true', help='Verify a frozen prepared archive without re-fitting or changing its bytes.')
    args = parser.parse_args(argv)
    for value in (args.pair_id, args.archive, args.config, args.prepared_archive, args.receipt, args.code_archive, args.input_paths):
        require_basename(value)
    if not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)+', args.pair_id):
        parser.error('pair-id must be an underscore-separated channel pair.')
    if Path(args.prepared_archive).exists() and not args.verify_existing:
        raise FileExistsError(f'Refusing to overwrite existing output archive: {args.prepared_archive}')
    details = {'schema_version': 1, 'stage': 'prepare', 'pair_id': args.pair_id,
               'success': False, 'synthetic': False, 'archive': args.prepared_archive,
               'config': args.config, 'code_archive': args.code_archive,
               'raw_archive': None if args.verify_existing else args.archive,
               'provenance_kind': 'verified_existing_prepared' if args.verify_existing else 'new_preparation'}
    try:
        details['config_sha256'] = sha256_file(args.config)
        details['code_archive_sha256'] = sha256_file(args.code_archive)
        details['source_archive_sha256'] = details['code_archive_sha256']
        details['input_paths_sha256'] = None if args.verify_existing else sha256_file(args.input_paths)
        write_json(args.receipt, details)
        if args.verify_existing:
            verify_existing(args, details)
        else:
            prepare_and_archive(args, details)
    except Exception as exc:
        details.update(success=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_json(args.receipt, details)
        raise


if __name__ == '__main__':
    main()
