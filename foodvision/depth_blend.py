"""Nested evaluation of optional depth blending; never select on outer labels."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import ROOT, verify_payload
from .depth import digest
from .diagnostics import summarize
from .features import meal_representations
from .models import CandidateRegressors
from .selection import evaluate_episode, group_mae
from .segmentation_experiment import PARAMETERS, check_boundaries, paired_block_interval, predict_candidates
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
POLICIES = ('Blend', 'BoundedDownward')

def combine(base, depth, available, weight, policy):
    base, depth, available = (np.asarray(base), np.asarray(depth), np.asarray(available, dtype=bool))
    if base.shape != depth.shape or available.shape != base.shape:
        raise ValueError('Experts and availability must be aligned.')
    if not np.isfinite(base).all() or not np.isfinite(depth).all() or (not 0 <= weight <= 1):
        raise ValueError('Expected finite predictions and a weight in [0,1].')
    if policy not in POLICIES:
        raise ValueError('Unknown depth policy.')
    delta = depth - base
    if policy == 'BoundedDownward':
        delta = np.clip(delta, -0.1, 0)
    return np.clip(base + weight * np.where(available, delta, 0), 0, 1.25)

def choose_weight(base, depth, available, targets, groups, policy):
    baseline = summarize(targets, base, groups)
    candidates, scores, admissible = ([], [], [])
    for weight in WEIGHTS:
        forecast = combine(base, depth, available, weight, policy)
        score = summarize(targets, forecast, groups)
        small_base = baseline['positive_up_to_10_percent']['group_mae_pp']
        small_new = score['positive_up_to_10_percent']['group_mae_pp']
        passes = (small_base is None or small_new <= small_base + 1) and score['recorded_zero']['predictions_above_10_percent_count'] <= baseline['recorded_zero']['predictions_above_10_percent_count']
        candidates.append(forecast)
        scores.append(score['all']['group_mae_pp'])
        admissible.append(passes)
    selected = min((i for i, ok in enumerate(admissible) if ok), key=lambda i: (scores[i], i))
    return (WEIGHTS[selected], {'weight': WEIGHTS[selected], 'inner_mae_pp': scores[selected], 'inner_baseline_mae_pp': scores[0], 'candidate_mae_pp': scores, 'admissible': admissible, 'inner_recorded_zero_count': baseline['recorded_zero']['records'], 'inner_small_positive_count': baseline['positive_up_to_10_percent']['records']})

def forecast(records, representations, depth_features, available, targets, groups, episode):
    original = evaluate_episode(records, episode, CandidateRegressors(representations, targets, groups))
    base_id = int(original.selected_candidates[1])
    base_inner = np.clip(original.inner_predictions[base_id], 0, 1.25)
    base_outer = np.clip(original.test_forecasts.predictions[base_id], 0, 1.25)
    train, test = (episode['train'], episode['test'])
    positions = {index: i for i, index in enumerate(train)}
    banks = []
    for features in depth_features.values():
        inner = np.full((len(PARAMETERS), len(train)), np.nan)
        visits = np.zeros(len(train), dtype=int)
        for fold in episode['inner']:
            loc = [positions[i] for i in fold['validation']]
            inner[:, loc] = predict_candidates(features, targets, groups, fold['train'], fold['validation'])
            visits[loc] += 1
        if not np.all(visits == 1) or not np.isfinite(inner).all():
            raise ValueError('Every fitting record requires exactly one inner forecast.')
        banks.append(np.clip(inner, 0, 1.25))
    joined = np.vstack(banks)
    depth_id = int(np.argmin(group_mae(joined, targets[train], groups[train])))
    family, candidate = divmod(depth_id, len(PARAMETERS))
    name = list(depth_features)[family]
    depth_inner = joined[depth_id]
    depth_outer = np.clip(predict_candidates(depth_features[name], targets, groups, train, test)[candidate], 0, 1.25)
    results = {'Change': base_outer}
    selection = {'base_candidate': base_id, 'depth_family': name, 'depth_candidate': candidate, 'outer_geometry_available_count': int(available[test].sum())}
    for policy in POLICIES:
        weight, details = choose_weight(base_inner, depth_inner, available[train], targets[train], groups[train], policy)
        results[policy] = combine(base_outer, depth_outer, available[test], weight, policy)
        selection[policy] = details
    return (results, selection)

def run(cache, mask_cache, output, root=ROOT):
    cache, mask_cache, output = map(Path, (cache, mask_cache, output))
    if output.exists():
        raise FileExistsError(output)
    verify_payload(root)
    manifest = json.loads((cache / 'manifest.json').read_text())
    masks_manifest = json.loads((mask_cache / 'manifest.json').read_text())
    if manifest['protocol_sha256'] != digest(root / 'docs/DEPTH-PROTOCOL.json'):
        raise ValueError('Depth extraction protocol changed.')
    if manifest['mask_manifest_sha256'] != digest(mask_cache / 'manifest.json'):
        raise ValueError('Mask provenance changed.')
    all_records = json.loads((root / 'data/records.json').read_text())
    global_ids = [i for i, r in enumerate(all_records) if r['source'] == 'LeFood']
    local = {g: i for i, g in enumerate(global_ids)}
    records = [all_records[i] for i in global_ids]
    ids = [r['record_id'] for r in records]
    matrices = []
    for directory, metadata in ((cache, manifest), (mask_cache, masks_manifest)):
        if metadata['features_sha256'] != digest(directory / 'features.npz'):
            raise ValueError('Feature checksum mismatch.')
        if [r['record_id'] for r in metadata['records']] != ids:
            raise ValueError('Provenance rows differ.')
        with np.load(directory / 'features.npz', allow_pickle=False) as data:
            if data['record_ids'].tolist() != ids:
                raise ValueError('Feature rows differ.')
            matrices.append(data['paired'].copy())
    depth, masks = matrices
    available = (depth[:, 1] == 1) & (depth[:, 10] == 1)
    with np.load(root / 'data/features.npz', allow_pickle=False) as data:
        if data['record_ids'].tolist() != [r['record_id'] for r in all_records]:
            raise ValueError('Embedding rows differ.')
        representations = meal_representations(data['before'][global_ids], data['after'][global_ids])
    change = representations['Cosine change']
    features = {'ChangeDepth': np.column_stack([change, depth]), 'ChangeMaskDepth': np.column_stack([change, masks, depth])}
    targets, groups, blocks = (np.array([r[k] for r in records]) for k in ('fraction', 'group', 'dependency_block'))
    refs = json.loads((root / 'data/reference_predictions.json').read_text())
    predictions = {k: np.full(len(records), np.nan) for k in ('Change', *POLICIES)}
    selections, visits = ([], np.zeros(len(records), dtype=int))
    with threadpool_limits(limits=2):
        for e in json.loads((root / 'data/episodes.json').read_text()):
            if e['source'] != 'LeFood':
                continue
            episode = {**e, 'train': [local[i] for i in e['train']], 'test': [local[i] for i in e['test']], 'inner': [{k: [local[i] for i in v] for k, v in f.items()} for f in e['inner']]}
            overlap = check_boundaries(records, manifest['records'], episode['train'], episode['test'])
            for f in episode['inner']:
                check_boundaries(records, manifest['records'], f['train'], f['validation'])
            outputs, selection = forecast(records, representations, features, available, targets, groups, episode)
            np.testing.assert_allclose(outputs['Change'], np.clip(refs[e['name']]['raw_predictions'][2], 0, 1.25), atol=1e-10, rtol=0)
            for name, p in outputs.items():
                predictions[name][episode['test']] = p
            visits[episode['test']] += 1
            selections.append({'episode': e['name'], 'broader_dependency_blocks_crossing_outer_boundary': overlap, **selection})
            print(f"Blended {e['name']}", flush=True)
    if not np.all(visits == 1) or not all((np.isfinite(p).all() for p in predictions.values())):
        raise ValueError('Missing or repeated outer forecasts.')
    summary = {k: summarize(targets, p, groups) for k, p in predictions.items()}
    comparisons = {}
    for policy in POLICIES:
        interval = paired_block_interval(targets, predictions[policy], predictions['Change'], groups, blocks)
        base, new = (summary['Change'], summary[policy])
        gates = {'overall_error_not_increased': new['all']['group_mae_pp'] <= base['all']['group_mae_pp'], 'small_positive_degradation_at_most_1_pp': new['positive_up_to_10_percent']['group_mae_pp'] <= base['positive_up_to_10_percent']['group_mae_pp'] + 1, 'zero_above_10_percent_count_not_increased': new['recorded_zero']['predictions_above_10_percent_count'] <= base['recorded_zero']['predictions_above_10_percent_count'], 'paired_interval_below_zero': interval['interval_95_pp'][1] < 0}
        by_group = [{'group': g, 'error_change_pp': float(100 * (np.abs(predictions[policy][groups == g] - targets[groups == g]) - np.abs(predictions['Change'][groups == g] - targets[groups == g])).mean())} for g in sorted(set(groups))]
        comparisons[policy] = {**interval, 'gates': gates, 'all_gates_pass': all(gates.values()), 'by_group': by_group}
    protocol_path = root / 'docs/DEPTH-BLEND-PROTOCOL.json'
    report = {'protocol': json.loads(protocol_path.read_text()), 'protocol_sha256': digest(protocol_path), 'code_sha256': digest(Path(__file__)), 'feature_manifest_sha256': digest(cache / 'manifest.json'), 'summary': summary, 'comparisons': comparisons, 'selections': selections, 'rows': [{**r, 'geometry_available': bool(available[i]), 'predictions': {k: float(p[i]) for k, p in predictions.items()}} for i, r in enumerate(records)]}
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
