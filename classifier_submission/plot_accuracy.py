#!/usr/bin/env python3
"""Plot existing classifier JSON results from one explicit batch; never train."""
import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import tarfile
import textwrap

PAIRS = {
    'ggf_ttbar': 'Single-Higgs (gluon fusion) vs. Top-quark pair',
    'ggf_dihiggs': 'Single-Higgs (gluon fusion) vs. Higgs pair',
    'ttbar_dihiggs': 'Top-quark pair vs. Higgs pair',
}
REPRESENTATIONS = {
    'general': {'label': 'General event features', 'dim': 64},
    'energy': {'label': 'Energy scale', 'dim': 64},
    'eta': {'label': 'Pseudorapidity energy profile', 'dim': 64},
    'phi': {'label': 'Azimuthal structure', 'dim': 64},
    'local': {'label': 'Multiscale energy correlations', 'dim': 64},
    'concat': {'label': 'Combined representation', 'dim': 320},
}
TRANSFORMS = {
    'r': ('rotate', 'Rotation'),
    'e': ('energy_noise', 'Energy noise'),
    'x': ('xyz_noise', 'Hit-position jitter'),
    's': ('shift', 'Global transverse shift'),
    'c': ('crop', 'Spatial masking'),
}
AUGMENTATIONS = {'none': ()}
for size in (3, 4, 5):
    for subset in combinations(TRANSFORMS, size):
        AUGMENTATIONS[''.join(subset)] = tuple(TRANSFORMS[key][0] for key in subset)
SPLITS = ('train', 'val', 'test')
SPLIT_LABELS = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
LONG_FIELDS = (
    'run_id', 'pair_id', 'task_label', 'augmentation', 'augmentation_label',
    'n_augmentations', 'representation', 'representation_label', 'embedding_dim',
    'split', 'accuracy', 'accuracy_percent', 'none_accuracy', 'delta_accuracy_pp',
    'prepared_fingerprint', 'source_file',
)


def augmentation_label(suffix):
    return ('No augmentation' if suffix == 'none'
            else ' + '.join(TRANSFORMS[key][1] for key in suffix))


def batch_path(root, relative):
    """Even symlinks may not silently select results from another batch."""
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Input leaves the specified classifier batch: {relative}')
    return path


def read_object(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path.name}')
    return value


def read_plan(root):
    plan = read_object(batch_path(root, 'evaluation_plan.json'))
    runs = plan.get('runs')
    if not isinstance(runs, list):
        raise ValueError('evaluation_plan.json must contain a runs list')
    expected = {f'{pair}_{aug}' for pair in PAIRS for aug in AUGMENTATIONS}
    records = {}
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get('run_id'), str):
            raise ValueError('Each plan run must have a run_id')
        run_id, pair = run['run_id'], run.get('pair_id')
        if (run_id not in expected or pair not in PAIRS
                or not run_id.startswith(pair + '_') or run_id in records):
            raise ValueError(f'Duplicate or mismatched run_id/pair_id in plan: {run_id}')
        records[run_id] = run
    if set(records) != expected:
        raise ValueError('Expected the 51 unique planned runs: three pairs with 17 combinations each')
    return records


def read_classification(root, run_id):
    status_relative = f'results/classifier_status_{run_id}.json'
    source = status_relative
    status_path = batch_path(root, status_relative)
    if not status_path.is_file():
        return None, source, 'missing', ['Status JSON is missing; archive fallback is not allowed without success']
    try:
        status = read_object(status_path)
        if status.get('run_id') != run_id:
            raise ValueError('Status run_id does not match the plan')
        if (status.get('success') is not True
                or type(status.get('exit_code')) is not int or status['exit_code'] != 0
                or type(status.get('archive_exit_code')) is not int or status['archive_exit_code'] != 0):
            raise ValueError('Status must have success=true, exit_code=0 and archive_exit_code=0')
        if 'classification' in status:
            classification = status['classification']
        else:
            # result_path in the plan refers to PRETRAINING. Never follow it.
            archive_relative = f'results/classifier_{run_id}.tar.gz'
            member_name = f'{run_id}/metrics.json'
            source = f'{archive_relative}::{member_name}'
            with tarfile.open(batch_path(root, archive_relative), 'r:gz') as archive:
                matches = [m for m in archive.getmembers() if m.name == member_name]
                if len(matches) != 1 or not matches[0].isfile():
                    raise ValueError('Archive must contain exactly one regular ' + member_name)
                with archive.extractfile(matches[0]) as stream:
                    metrics = json.load(stream)
                if not isinstance(metrics, dict) or 'classification' not in metrics:
                    raise ValueError('Archived metrics.json lacks classification')
                classification = metrics['classification']
        if not isinstance(classification, dict):
            raise ValueError('classification must be an object')
        return classification, source, 'success', []
    except (OSError, ValueError, tarfile.TarError, EOFError) as exc:
        return None, source, 'failed', [str(exc)]


def accuracy_value(classification, representation, split):
    try:
        value = classification[representation][split]['accuracy']
    except (KeyError, TypeError):
        raise ValueError(f'{representation}.{split}.accuracy is missing') from None
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{representation}.{split}.accuracy must be a finite number in [0,1]')
    return value


def collect_accuracy(classifier_dir):
    """Read only the plan and this batch's status/necessary metrics JSON files."""
    root = Path(classifier_dir).expanduser().resolve()
    planned = read_plan(root)
    conflicting_pairs = set()
    for pair in PAIRS:
        fingerprints = {record.get('prepared_fingerprint') for record in planned.values()
                        if record['pair_id'] == pair and isinstance(record.get('prepared_fingerprint'), str)
                        and record['prepared_fingerprint'].strip()}
        if len(fingerprints) > 1:
            conflicting_pairs.add(pair)
    loaded, available_splits = {}, set()
    for pair in PAIRS:
        for aug in AUGMENTATIONS:
            run_id = f'{pair}_{aug}'
            try:
                classification, source, state, reasons = read_classification(root, run_id)
            except ValueError as exc:
                classification, source, state, reasons = None, '', 'failed', [str(exc)]
            if classification is not None:
                for rep in REPRESENTATIONS:
                    metrics = classification.get(rep)
                    if isinstance(metrics, dict):
                        available_splits.update(split for split in SPLITS if split in metrics)
            loaded[run_id] = (classification, source, state, reasons)
    # With no readable scores, preserve the full expected grid of NA rows.
    splits = [split for split in SPLITS if split in available_splits] or list(SPLITS)
    rows, run_reports = [], []
    for pair in PAIRS:
        for aug in AUGMENTATIONS:
            run_id = f'{pair}_{aug}'
            classification, source, state, reasons = loaded[run_id]
            reasons = list(reasons)
            fingerprint = planned[run_id].get('prepared_fingerprint')
            if not isinstance(fingerprint, str) or not fingerprint.strip():
                reasons.append('Plan prepared_fingerprint is missing or invalid')
                classification, state, fingerprint = None, 'failed', None
            if pair in conflicting_pairs:
                reasons.append('Plan prepared_fingerprint differs within this pair; pair scores withheld')
                classification, state = None, 'failed'
            for rep, display in REPRESENTATIONS.items():
                for split in splits:
                    accuracy = None
                    if classification is not None:
                        try:
                            accuracy = accuracy_value(classification, rep, split)
                        except ValueError as exc:
                            reasons.append(str(exc))
                            state = 'failed'
                    rows.append({
                        'run_id': run_id, 'pair_id': pair, 'task_label': PAIRS[pair],
                        'augmentation': aug, 'augmentation_label': augmentation_label(aug),
                        'n_augmentations': len(AUGMENTATIONS[aug]),
                        'representation': rep, 'representation_label': display['label'],
                        'embedding_dim': display['dim'], 'split': split, 'accuracy': accuracy,
                        'accuracy_percent': 100 * accuracy if accuracy is not None else None,
                        'none_accuracy': None, 'delta_accuracy_pp': None,
                        'prepared_fingerprint': fingerprint, 'source_file': source,
                    })
            run_reports.append({'run_id': run_id, 'pair_id': pair, 'status': state,
                                'source_file': source, 'reasons': reasons})
    none_values = {(r['pair_id'], r['representation'], r['split']): r['accuracy']
                   for r in rows if r['augmentation'] == 'none'}
    for row in rows:
        baseline = none_values[(row['pair_id'], row['representation'], row['split'])]
        row['none_accuracy'] = baseline
        if row['accuracy'] is not None and baseline is not None:
            row['delta_accuracy_pp'] = 100 * (row['accuracy'] - baseline)
    counts = Counter(record['status'] for record in run_reports)
    report = {
        'classifier_dir': str(root), 'plan_file': 'evaluation_plan.json',
        'counts': {'expected': len(planned), **{key: counts[key] for key in ('success', 'failed', 'missing')}},
        'available_splits': [split for split in SPLITS if split in available_splits],
        'table_splits': splits,
        'valid_accuracy_counts': {split: sum(r['split'] == split and r['accuracy'] is not None for r in rows)
                                  for split in SPLITS},
        'prepared_fingerprint_source': 'evaluation_plan.json runs; consistency only, not revalidation of prepared data',
        'runs': run_reports,
    }
    return rows, report


def plot_dependencies():
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Plotting requires NumPy and Matplotlib. Use an existing Python environment '
                           'containing numpy and matplotlib; nothing will be installed automatically.') from exc
    return np, plt


def base_figure(plt, pair, split, title, valid, total, *, heatmap=False):
    fig = plt.figure(figsize=(16 if heatmap else 13, 18.5))
    ax = fig.add_axes([0.33 if heatmap else 0.39, 0.105, 0.58 if heatmap else 0.57,
                       0.76 if heatmap else 0.79])
    fig.suptitle(title, x=0.5, y=0.976, fontsize=17)
    fig.text(0.5, 0.951, f'{PAIRS[pair]} | {SPLIT_LABELS[split]}', ha='center', fontsize=12)
    fig.text(0.5, 0.931, f'Valid values: {valid}/{total} | Pretraining augmentations', ha='center', fontsize=11)
    fig.text(0.5, 0.035, 'Rows grouped by augmentation count: none (1), three (10), four (5), five (1).',
             ha='center', fontsize=10)
    labels = [augmentation_label(aug).replace(' + ', ' +\n') for aug in AUGMENTATIONS]
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    for boundary in (0.5, 10.5, 15.5):
        ax.axhline(boundary, color='0.55', linewidth=0.8, linestyle='--')
    return fig, ax


def create_figures(rows, split):
    """Return nine independent Matplotlib figures; the caller must close them."""
    if split not in SPLITS:
        raise ValueError('split must be test, val or train')
    np, plt = plot_dependencies()
    lookup = {(r['pair_id'], r['augmentation'], r['representation']): r
              for r in rows if r['split'] == split}
    def value(pair, aug, rep, field):
        result = lookup.get((pair, aug, rep), {}).get(field)
        return np.nan if result is None else result
    all_deltas = [value(pair, aug, 'concat', 'delta_accuracy_pp') for pair in PAIRS for aug in AUGMENTATIONS]
    max_delta = max((abs(v) for v in all_deltas if math.isfinite(v)), default=0)
    delta_limit = max(1.0, max_delta * 1.2)
    figures = []
    for pair in PAIRS:
        accuracy = np.array([value(pair, aug, 'concat', 'accuracy_percent') for aug in AUGMENTATIONS])
        fig, ax = base_figure(plt, pair, split, 'Classification accuracy — Combined representation',
                              int(np.isfinite(accuracy).sum()), 17)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Accuracy (%)', fontsize=11, labelpad=12)
        ax.grid(axis='x', alpha=0.25)
        reference = accuracy[0]
        if np.isfinite(reference):
            ax.axvline(reference, linestyle='--', color='0.35', linewidth=1,
                       label=f'No augmentation: {reference:.1f}%')
            ax.legend(loc='lower right', bbox_to_anchor=(1, 1.005), fontsize=10, frameon=False)
        else:
            ax.text(1, 1.01, 'No augmentation reference: NA', transform=ax.transAxes, ha='right', fontsize=10)
        for index, score in enumerate(accuracy):
            if np.isfinite(score):
                ax.scatter(score, index, color='C0', zorder=3)
                left = score > 90
                ax.annotate(f'{score:.1f}%', (score, index), xytext=(-8 if left else 8, 0),
                            textcoords='offset points', va='center', ha='right' if left else 'left', fontsize=10)
            else:
                ax.text(2, index, 'NA', va='center', color='0.4', fontsize=10)
        figures.append((f'concat_accuracy_{pair}_{split}', fig))

        matrix = np.array([[value(pair, aug, rep, 'accuracy_percent') for rep in REPRESENTATIONS]
                           for aug in AUGMENTATIONS])
        fig, ax = base_figure(plt, pair, split, 'Classification accuracy by representation',
                              int(np.isfinite(matrix).sum()), 102, heatmap=True)
        cmap = plt.get_cmap().copy()
        cmap.set_bad('0.94')
        artist = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap=cmap, aspect='auto')
        ax.set_xticks(range(6), ['\n'.join(textwrap.wrap(d['label'], width=22)) + f"\n({d['dim']}D)"
                                 for d in REPRESENTATIONS.values()], fontsize=10)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', length=0, pad=12)
        for row in range(17):
            for col in range(6):
                score = matrix[row, col]
                ax.text(col, row, f'{score:.1f}' if np.isfinite(score) else 'NA',
                        ha='center', va='center', fontsize=10,
                        color='white' if np.isfinite(score) and score < 50 else 'black')
        color_axis = fig.add_axes([0.93, 0.105, 0.015, 0.76])
        fig.colorbar(artist, cax=color_axis, label='Accuracy (%)')
        figures.append((f'space_accuracy_{pair}_{split}', fig))

        delta = np.array([value(pair, aug, 'concat', 'delta_accuracy_pp') for aug in AUGMENTATIONS])
        fig, ax = base_figure(plt, pair, split, 'Accuracy change relative to no augmentation',
                              int(np.isfinite(delta).sum()), 17)
        ax.set_xlim(-delta_limit, delta_limit)
        ax.axvline(0, color='0.35', linewidth=1)
        ax.set_xlabel('Accuracy difference vs no augmentation (percentage points)', fontsize=11, labelpad=12)
        ax.grid(axis='x', alpha=0.25)
        for index, change in enumerate(delta):
            if np.isfinite(change):
                ax.scatter(change, index, color='C0', zorder=3)
                ax.annotate(f'{change:+.1f}', (change, index), xytext=(8 if change >= 0 else -8, 0),
                            textcoords='offset points', va='center', ha='left' if change >= 0 else 'right', fontsize=10)
            else:
                ax.text(delta_limit * 0.05, index, 'NA', va='center', color='0.4', fontsize=10)
        figures.append((f'concat_delta_{pair}_{split}', fig))
    return figures


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: 'NA' if row.get(field) is None else row[field] for field in fields})


def write_outputs(classifier_dir, split='test', output_dir=None):
    rows, report = collect_accuracy(classifier_dir)
    output = Path(output_dir).expanduser().resolve() if output_dir else Path(report['classifier_dir']) / 'visualization'
    figures = create_figures(rows, split)
    _, plt = plot_dependencies()
    try:
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'accuracy_long.csv', LONG_FIELDS, rows)
        identifiers = ('run_id', 'pair_id', 'task_label', 'augmentation', 'augmentation_label',
                       'n_augmentations', 'prepared_fingerprint')
        percent_fields = [f'{rep}_accuracy_percent' for rep in REPRESENTATIONS]
        by_id = {row['run_id']: row for row in rows}
        by_value = {(r['run_id'], r['representation']): r['accuracy_percent'] for r in rows if r['split'] == split}
        wide = []
        for pair in PAIRS:
            for aug in AUGMENTATIONS:
                run_id = f'{pair}_{aug}'
                row = {field: by_id[run_id][field] for field in identifiers}
                row.update({f'{rep}_accuracy_percent': by_value.get((run_id, rep)) for rep in REPRESENTATIONS})
                wide.append(row)
        write_csv(output / f'accuracy_{split}_wide.csv', (*identifiers, *percent_fields), wide)
        written = ['accuracy_long.csv', f'accuracy_{split}_wide.csv', 'read_report.json']
        for name, figure in figures:
            for extension in ('png', 'pdf'):
                path = output / f'{name}.{extension}'
                figure.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.2)
                written.append(path.name)
            plt.close(figure)
        report.update(output_dir=str(output), plotted_split=split, outputs=written)
        (output / 'read_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    finally:
        for _, figure in figures:
            plt.close(figure)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier-dir', required=True, help='One batch containing evaluation_plan.json and results/')
    parser.add_argument('--split', choices=('test', 'val', 'train'), default='test')
    parser.add_argument('--output-dir', help='Defaults to CLASSIFIER_DIR/visualization/')
    args = parser.parse_args(argv)
    try:
        report = write_outputs(args.classifier_dir, args.split, args.output_dir)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        parser.exit(1, f'Cannot plot classifier accuracy: {exc}\n')
    counts = report['counts']
    print('Classifier directory:', report['classifier_dir'])
    print(f"Runs: expected={counts['expected']}, success={counts['success']}, failed={counts['failed']}, missing={counts['missing']}")
    print('Valid accuracy values:', ', '.join(f'{split}={report["valid_accuracy_counts"][split]}' for split in SPLITS))
    print(f"Output: {report['output_dir']} (9 PNG + 9 PDF, 2 CSV, read_report.json; split={args.split})")
    print('Missing/invalid scores remain NA; prepared fingerprints are plan identifiers, not revalidated data.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
