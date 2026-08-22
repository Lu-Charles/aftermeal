"""Matched nested evaluation of relative-depth features and no-depth controls."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import ROOT, verify_payload
from .depth import digest
from .diagnostics import summarize
from .features import meal_representations
from .segmentation_experiment import check_boundaries, nested_forecast, paired_block_interval

def run(cache, mask_cache, output, root=ROOT):
    cache, mask_cache, output = map(Path, (cache, mask_cache, output))
    if output.exists():
        raise FileExistsError(output)
    verify_payload(root)
    protocol_path = root / 'docs/DEPTH-PROTOCOL.json'
    manifest = json.loads((cache / 'manifest.json').read_text())
    mask_manifest = json.loads((mask_cache / 'manifest.json').read_text())
    if manifest['protocol_sha256'] != digest(protocol_path):
        raise ValueError('Depth protocol changed after extraction.')
    if manifest['mask_manifest_sha256'] != digest(mask_cache / 'manifest.json'):
        raise ValueError('Mask provenance changed after extraction.')
    for path, expected in ((cache / 'features.npz', manifest['features_sha256']), (mask_cache / 'features.npz', mask_manifest['features_sha256'])):
        if digest(path) != expected:
            raise ValueError(f'Feature cache changed: {path}')
    all_records = json.loads((root / 'data/records.json').read_text())
    global_ids = [i for i, r in enumerate(all_records) if r['source'] == 'LeFood']
    local = {g: i for i, g in enumerate(global_ids)}
    records = [all_records[i] for i in global_ids]
    ids = [r['record_id'] for r in records]
    for metadata in (manifest, mask_manifest):
        if [r['record_id'] for r in metadata['records']] != ids:
            raise ValueError('Provenance rows do not match labels.')
    for a, b in zip(manifest['records'], mask_manifest['records']):
        for role in ('before', 'after'):
            if a[role]['sha256'] != b[role]['sha256']:
                raise ValueError('Depth and masks describe different photographs.')
    matrices = []
    for directory in (cache, mask_cache):
        with np.load(directory / 'features.npz', allow_pickle=False) as data:
            if data['record_ids'].tolist() != ids:
                raise ValueError('Feature rows do not match labels.')
            matrices.append(data['paired'].copy())
    depth, masks = matrices
    with np.load(root / 'data/features.npz', allow_pickle=False) as data:
        if data['record_ids'].tolist() != [r['record_id'] for r in all_records]:
            raise ValueError('Embedding rows do not match labels.')
        change = meal_representations(data['before'][global_ids], data['after'][global_ids])['Cosine change']
    features = {'ChangeControl': change, 'ChangeMask': np.column_stack([change, masks]), 'ChangeDepth': np.column_stack([change, depth]), 'ChangeMaskDepth': np.column_stack([change, masks, depth])}
    if not all((np.isfinite(x).all() for x in features.values())):
        raise ValueError('Non-finite experiment features.')
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    blocks = np.array([r['dependency_block'] for r in records])
    predictions = {name: np.full(len(records), np.nan) for name in ('Change', *features)}
    refs = json.loads((root / 'data/reference_predictions.json').read_text())
    selections, visits = ([], np.zeros(len(records), dtype=int))
    with threadpool_limits(limits=2):
        for e in json.loads((root / 'data/episodes.json').read_text()):
            if e['source'] != 'LeFood':
                continue
            episode = {**e, 'train': [local[i] for i in e['train']], 'test': [local[i] for i in e['test']], 'inner': [{k: [local[i] for i in indices] for k, indices in f.items()} for f in e['inner']]}
            overlap = check_boundaries(records, manifest['records'], episode['train'], episode['test'])
            for fold in episode['inner']:
                check_boundaries(records, manifest['records'], fold['train'], fold['validation'])
            predictions['Change'][episode['test']] = np.clip(refs[e['name']]['raw_predictions'][2], 0, 1.25)
            selected = {'episode': e['name'], 'broader_dependency_blocks_crossing_outer_boundary': overlap}
            for name, x in features.items():
                predictions[name][episode['test']], selected[name] = nested_forecast(x, targets, groups, episode)
            visits[episode['test']] += 1
            selections.append(selected)
            print(f"Evaluated depth {e['name']}", flush=True)
    if not np.all(visits == 1) or not all((np.isfinite(p).all() for p in predictions.values())):
        raise ValueError('Each row requires exactly one finite held-out prediction.')
    summary = {k: summarize(targets, p, groups) for k, p in predictions.items()}
    comparisons = {}
    for name in ('ChangeDepth', 'ChangeMaskDepth'):
        matched = 'ChangeMask' if name == 'ChangeMaskDepth' else 'ChangeControl'
        base, new = (summary['Change'], summary[name])
        interval = paired_block_interval(targets, predictions[name], predictions['Change'], groups, blocks)
        control_interval = paired_block_interval(targets, predictions[name], predictions[matched], groups, blocks)
        gates = {'MAE_reduction_at_least_10_percent': new['all']['group_mae_pp'] <= 0.9 * base['all']['group_mae_pp'], 'small_positive_degradation_at_most_1_pp': new['positive_up_to_10_percent']['group_mae_pp'] <= base['positive_up_to_10_percent']['group_mae_pp'] + 1, 'no_more_recorded_zero_predictions_above_10_percent': new['recorded_zero']['predictions_above_10_percent_count'] <= base['recorded_zero']['predictions_above_10_percent_count'], 'paired_interval_below_zero': interval['interval_95_pp'][1] < 0, 'better_MAE_than_no_depth_control': new['all']['group_mae_pp'] < summary[matched]['all']['group_mae_pp']}
        comparisons[name] = {'against_original': interval, 'no_depth_control': matched, 'against_no_depth_control': control_interval, 'gates': gates, 'all_gates_pass': all(gates.values())}
    report = {'protocol': json.loads(protocol_path.read_text()), 'protocol_sha256': digest(protocol_path), 'evaluation_amendment': json.loads((root / 'docs/DEPTH-EVAL-AMENDMENT.json').read_text()), 'evaluation_amendment_sha256': digest(root / 'docs/DEPTH-EVAL-AMENDMENT.json'), 'feature_manifest_sha256': digest(cache / 'manifest.json'), 'evaluation_code_sha256': digest(Path(__file__)), 'summary': summary, 'comparisons': comparisons, 'selections': selections, 'visual_residue_accuracy': 'Unmeasured: no independently verified visual labels. Zero-mass slices are not residue labels.', 'rows': [{**r, 'predictions': {k: float(p[i]) for k, p in predictions.items()}} for i, r in enumerate(records)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--mask-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cache, args.mask_cache, args.output)
    print(json.dumps({k: v['all']['group_mae_pp'] for k, v in report['summary'].items()}, indent=2))
