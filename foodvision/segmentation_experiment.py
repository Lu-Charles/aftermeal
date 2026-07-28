"""Nested grouped comparison of frozen mask features and the original baseline."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import ROOT, verify_payload
from .diagnostics import summarize
from .features import meal_representations
from .models import CandidateRegressors, GAMMAS, PENALTIES, group_weights, lower_weighted_median
from .selection import check_partition, evaluate_episode, group_mae
PARAMETERS = [(None, None)] + [(g, a) for g in GAMMAS for a in PENALTIES]

def standardized(train, test):
    """Fit every normalization statistic on this training partition only."""
    train, test = (np.asarray(train, dtype=float), np.asarray(test, dtype=float))
    mean, scale = (train.mean(axis=0), train.std(axis=0))
    scale[scale < 1e-12] = 1
    return ((train - mean) / scale / np.sqrt(train.shape[1]), (test - mean) / scale / np.sqrt(train.shape[1]))

def predict_candidates(features, targets, groups, train, test):
    if not train:
        return np.zeros((len(PARAMETERS), len(test)))
    x, z = standardized(features[train], features[test])
    weights = group_weights(groups[train])
    offset = lower_weighted_median(targets[train], weights)
    train_distance = np.maximum((x * x).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2 * x @ x.T, 0)
    test_distance = np.maximum((z * z).sum(1)[:, None] + (x * x).sum(1)[None, :] - 2 * z @ x.T, 0)
    predictions = [np.full(len(test), offset)]
    for gamma, alpha in PARAMETERS[1:]:
        kernel = np.exp(-gamma * train_distance)
        duals = np.linalg.solve(kernel + np.diag(alpha / weights), targets[train] - offset)
        predictions.append(np.exp(-gamma * test_distance) @ duals + offset)
    return np.asarray(predictions)

def check_boundaries(records, provenance, train, held_out):
    check_partition(records, train, held_out)

    def images(ids):
        return {provenance[i][kind]['sha256'] for i in ids for kind in ('before', 'after')}
    if images(train) & images(held_out):
        raise ValueError('An original image appears on both sides of a fitting boundary.')
    return sorted({records[i]['dependency_block'] for i in train} & {records[i]['dependency_block'] for i in held_out})

def nested_forecast(features, targets, groups, episode):
    train = episode['train']
    position = {i: j for j, i in enumerate(train)}
    validation = np.full((len(PARAMETERS), len(train)), np.nan)
    visits = np.zeros(len(train), dtype=int)
    for fold in episode['inner']:
        if not set(fold['train'] + fold['validation']).issubset(position):
            raise ValueError('Inner folds must be within the calibration records.')
        positions = [position[i] for i in fold['validation']]
        validation[:, positions] = predict_candidates(features, targets, groups, fold['train'], fold['validation'])
        visits[positions] += 1
    if not np.all(visits == 1) or not np.isfinite(validation).all():
        raise ValueError('Every calibration record needs exactly one finite inner forecast.')
    scores = group_mae(validation, targets[train], groups[train])
    chosen = int(np.argmin(scores))
    predictions = predict_candidates(features, targets, groups, train, episode['test'])[chosen]
    return (np.clip(predictions, 0, 1.25), {'candidate': chosen, 'gamma': PARAMETERS[chosen][0], 'alpha': PARAMETERS[chosen][1], 'inner_mae_pp': float(scores[chosen] * 100)})

def paired_block_interval(targets, prediction, baseline, groups, blocks, repeats=2000):
    """Resample dependency blocks, retaining equal total weight per food group."""
    group_names = sorted(set(groups))
    deltas, group_blocks = ([], [])
    for group in group_names:
        ids = groups == group
        membership = set(blocks[ids])
        if len(membership) != 1:
            raise ValueError('An original group spans dependency blocks.')
        group_blocks.append(next(iter(membership)))
        deltas.append(float((np.abs(prediction[ids] - targets[ids]) - np.abs(baseline[ids] - targets[ids])).mean() * 100))
    unique_blocks = sorted(set(group_blocks))
    block_sums = np.array([sum((d for d, b in zip(deltas, group_blocks) if b == block)) for block in unique_blocks])
    block_counts = np.array([group_blocks.count(block) for block in unique_blocks])
    rng = np.random.default_rng(20261003)
    samples = rng.integers(0, len(unique_blocks), size=(repeats, len(unique_blocks)))
    draws = block_sums[samples].sum(axis=1) / block_counts[samples].sum(axis=1)
    return {'difference_pp': float(np.mean(deltas)), 'interval_95_pp': np.quantile(draws, [0.025, 0.975]).tolist(), 'blocks': len(unique_blocks), 'resamples': repeats, 'scope': 'Conditional development interval for saved outer forecasts; does not include new training runs, label error or institution shift.'}

def run(cache: Path, output: Path, root: Path=ROOT):
    if output.exists():
        raise FileExistsError(f'Choose a new output path: {output}')
    verify_payload(root)
    protocol_bytes = (root / 'docs/SEGMENTATION-PROTOCOL.json').read_bytes()
    amendment_bytes = (root / 'docs/SEGMENTATION-AMENDMENT.json').read_bytes()
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['protocol_sha256'] != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError('Experiment protocol changed after feature extraction.')
    if manifest['features_sha256'] != hashlib.sha256((cache / 'features.npz').read_bytes()).hexdigest():
        raise ValueError('Segmentation feature cache changed.')
    all_records = json.loads((root / 'data/records.json').read_text())
    global_ids = [i for i, r in enumerate(all_records) if r['source'] == 'LeFood']
    records = [all_records[i] for i in global_ids]
    local = {global_id: i for i, global_id in enumerate(global_ids)}
    references = json.loads((root / 'data/reference_predictions.json').read_text())
    episodes = []
    for e in json.loads((root / 'data/episodes.json').read_text()):
        if e['source'] == 'LeFood':
            episodes.append({**e, 'train': [local[i] for i in e['train']], 'test': [local[i] for i in e['test']], 'inner': [{k: [local[i] for i in ids] for k, ids in fold.items()} for fold in e['inner']]})
    ids = [r['record_id'] for r in records]
    if [p['record_id'] for p in manifest['records']] != ids:
        raise ValueError('Provenance and label rows differ.')
    with np.load(cache / 'features.npz', allow_pickle=False) as data:
        if data['record_ids'].tolist() != ids:
            raise ValueError('Segmentation feature and label rows differ.')
        mask_features = data['paired'].copy()
    with np.load(root / 'data/features.npz', allow_pickle=False) as data:
        if data['record_ids'].tolist() != [r['record_id'] for r in all_records]:
            raise ValueError('Image embedding and label rows differ.')
        representations = meal_representations(data['before'][global_ids], data['after'][global_ids])
    features = {'Segmentation': mask_features, 'Combined': np.column_stack([representations['Cosine change'], mask_features])}
    if not all((np.isfinite(x).all() for x in features.values())):
        raise ValueError('All feature rows must be finite.')
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    blocks = np.array([r['dependency_block'] for r in records])
    predictions = {name: np.full(len(records), np.nan) for name in ('Change', 'Appearance', *features)}
    visits = np.zeros(len(records), dtype=int)
    selections = []
    with threadpool_limits(limits=2):
        regressors = CandidateRegressors(representations, targets, groups)
        for e in episodes:
            block_overlap = check_boundaries(records, manifest['records'], e['train'], e['test'])
            for fold in e['inner']:
                check_boundaries(records, manifest['records'], fold['train'], fold['validation'])
            original = evaluate_episode(records, e, regressors)
            original_ids = [0, int(original.selected_candidates[0]), int(original.selected_candidates[1])]
            if original_ids != references[e['name']]['candidate_ids']:
                raise ValueError('Original baseline selection changed.')
            np.testing.assert_allclose(original.test_forecasts.predictions[original_ids], references[e['name']]['raw_predictions'], atol=1e-10, rtol=0)
            predictions['Appearance'][e['test']] = np.clip(original.test_forecasts.predictions[original_ids[1]], 0, 1.25)
            predictions['Change'][e['test']] = np.clip(original.test_forecasts.predictions[original_ids[2]], 0, 1.25)
            selected = {'episode': e['name'], 'original_candidate_ids': original_ids, 'broader_dependency_blocks_crossing_outer_boundary': block_overlap}
            for name, x in features.items():
                predictions[name][e['test']], selected[name] = nested_forecast(x, targets, groups, e)
            selected['test_group_mae_pp'] = {k: float(group_mae(v[e['test']][None], targets[e['test']], groups[e['test']])[0] * 100) for k, v in predictions.items()}
            selections.append(selected)
            visits[e['test']] += 1
            print(f"Evaluated {e['name']}", flush=True)
    if not np.all(visits == 1) or not all((np.isfinite(p).all() for p in predictions.values())):
        raise ValueError('Every record needs exactly one finite outer forecast.')
    summary = {k: summarize(targets, p, groups) for k, p in predictions.items()}
    comparison = {}
    for name in features:
        interval = paired_block_interval(targets, predictions[name], predictions['Change'], groups, blocks)
        baseline, new = (summary['Change'], summary[name])
        gates = {'MAE_reduction_at_least_20_percent': new['all']['group_mae_pp'] <= 0.8 * baseline['all']['group_mae_pp'], 'small_positive_degradation_at_most_1_pp': new['positive_up_to_10_percent']['group_mae_pp'] <= baseline['positive_up_to_10_percent']['group_mae_pp'] + 1, 'halve_recorded_zero_predictions_above_10_percent': new['recorded_zero']['predictions_above_10_percent_count'] <= 0.5 * baseline['recorded_zero']['predictions_above_10_percent_count'], 'paired_interval_entirely_below_zero': interval['interval_95_pp'][1] < 0}
        comparison[name] = {**interval, 'provisional_gates': gates, 'all_provisional_gates_pass': all(gates.values())}
    report = {'protocol': json.loads(protocol_bytes), 'protocol_sha256': manifest['protocol_sha256'], 'amendment': json.loads(amendment_bytes), 'amendment_sha256': hashlib.sha256(amendment_bytes).hexdigest(), 'feature_manifest_sha256': hashlib.sha256((cache / 'manifest.json').read_bytes()).hexdigest(), 'records': len(records), 'summary': summary, 'comparison_to_Change': comparison, 'selections': selections, 'mask_quality': 'No independently verified masks available; IoU and mask confidence are not established.', 'label_review': 'No relabeling or exclusions. AI review proposals are not gold labels.', 'rows': [{**record, 'predictions': {k: float(p[i]) for k, p in predictions.items()}} for i, record in enumerate(records)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
