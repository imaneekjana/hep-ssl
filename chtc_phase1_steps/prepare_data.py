#!/usr/bin/env python3
"""Select unambiguous local train calo-hit shards and call the EXISTING preparation.
No changes to physics, event IDs, split logic, target definitions, or model code.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def discover_files(root, channel, pileup, override=None):
    root = Path(root).resolve()
    token = f'{channel}_{pileup}_calo_hits'
    if override:
        loc = (root / override).resolve()
        if not loc.is_relative_to(root):
            raise ValueError('input_paths.json paths must stay within the extracted raw/ directory.')
        files = list(loc.rglob('*.parquet')) if loc.is_dir() else [loc]
        files = [p for p in files if p.is_file() and p.suffix == '.parquet']
    else:
        # Exact configuration-directory component, not a substring channel guess.
        files = [p for p in root.rglob('*.parquet')
                 if token in p.relative_to(root).parts
                 and re.match(r'^train(?:[-_.]|$)', p.stem)
                 and p.is_file()]
    files = sorted(set(p.resolve() for p in files))
    if not files:
        examples = [str(p.relative_to(root)) for p in root.rglob('*.parquet')][:30]
        raise ValueError(f'No verified train calo-hit files for {token}. '
                         f'Actual Parquet examples: {examples}. '
                         'Set input_paths.json to the exact per-channel directory shown in this log.')
    parents = {p.parent for p in files}
    if not override and len(parents) != 1:
        raise ValueError(f'Multiple physical shard directories match {token}: '
                         f'{sorted(map(str, parents))}. Select one explicitly in input_paths.json.')
    if any(not p.is_relative_to(root) for p in files):
        raise ValueError('A dataset symlink resolves outside raw/. Supply a self-contained archive.')
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--input-paths', required=True)
    args = ap.parse_args()
    from src.config import load_config
    from src.data.events import file_hash
    from src.prepare_pairwise import prepare
    import polars as pl

    cfg = load_config(args.config)
    overrides = json.loads(Path(args.input_paths).read_text())
    if not isinstance(overrides, dict) or set(overrides) - set(cfg['data']['channels']):
        raise ValueError('input_paths.json must map configured channel names to raw-relative paths.')
    archive_hash = file_hash(args.archive)
    supplied_revision = cfg['data'].get('dataset_revision')
    if supplied_revision is None:
        cfg['data']['dataset_revision'] = 'local-cache-sha256:' + archive_hash
    revision_note = ('user-supplied upstream/local revision, retained unchanged' if supplied_revision
                     else 'exact local raw-archive SHA256; NOT a claim about upstream Hugging Face revision')
    print('Dataset snapshot:', cfg['data']['dataset_revision'], flush=True)
    print('Snapshot meaning:', revision_note, flush=True)
    source_records, inputs = {}, {}
    required = {'event_id', 'x', 'y', 'z', 'total_energy'}
    for channel in cfg['data']['channels']:
        files = discover_files(args.raw_root, channel, cfg['data']['pileup'], overrides.get(channel))
        folder = Path('selected_inputs') / channel
        folder.mkdir(parents=True, exist_ok=False)
        for i, path in enumerate(files):
            schema = pl.read_parquet_schema(path)
            if not required.issubset(schema):
                raise ValueError(f'{path} lacks {sorted(required - set(schema))}. No IDs will be fabricated.')
            (folder / f'{i:06d}_{path.name}').symlink_to(path)
        inputs[channel] = str(folder.resolve())
        source_records[channel] = [str(p.relative_to(Path(args.raw_root).resolve())) for p in files]
        print(f'{channel}: {len(files)} verified train shards', flush=True)
        for item in source_records[channel][:5]:
            print(' ', item, flush=True)
    details = {'raw_archive_sha256': archive_hash, 'dataset_revision': cfg['data']['dataset_revision'],
               'revision_meaning': revision_note, 'selected_source_files': source_records,
               'requested_channels': cfg['data']['channels'],
               'requested_events_per_channel': cfg['data']['events_per_channel']}
    Path('prepare_details.json').write_text(json.dumps(details, indent=2) + '\n')
    # Uses the source implementation and its exact target/split/statistics protocol.
    prepared = prepare(cfg, 'prepared', inputs=inputs, synthetic=False)
    m = prepared.metadata
    counts = Counter((r['channel'], r['split'], r['usable']) for r in m['manifest'])
    details.update({'success': True, 'synthetic': m['synthetic'], 'events': len(prepared.events),
                    'excluded_events': m['excluded_event_count'], 'manifest_hash': m['manifest_hash'],
                    'metadata_hash': m['metadata_hash'], 'grid_shape': list(prepared.grid.shape),
                    'eta_range': [prepared.grid.eta_edges[0], prepared.grid.eta_edges[-1]],
                    'range_source': prepared.grid.range_source,
                    'counts': [{'channel': k[0], 'split': k[1], 'usable': k[2], 'count': v}
                               for k, v in sorted(counts.items())],
                    'prepared_uncompressed_bytes': sum(p.stat().st_size for p in Path('prepared').rglob('*') if p.is_file())})
    Path('prepare_details.json').write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in details.items() if k != 'selected_source_files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
