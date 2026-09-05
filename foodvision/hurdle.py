"""Development experiment: separate recorded zero mass and positive amounts."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import ROOT, verify_payload
from .diagnostics import summarize
from .features import meal_representations
from .models import CandidateRegressors, GAMMAS, PENALTIES, group_weights, lower_weighted_median
from .selection import check_partition, group_mae
from .segmentation_experiment import paired_block_interval
PARAMETERS = [(feature, gamma, alpha) for feature in ('After', 'Paired', 'Cosine change') for gamma in GAMMAS for alpha in PENALTIES]

def predictions(kernels, targets, groups, train, test):
    if not train:
        return np.zeros((len(PARAMETERS), len(test)))
    positive = [i for i in train if targets[i] > 0]
    binary = (targets > 0).astype(float)
    cache = {}

    def ridge(feature, gamma, alpha, ids, y, bounds, stage):
        key = (feature, gamma, alpha, stage)
        if key not in cache:
            if not ids:
                cache[key] = np.zeros(len(test))
            else:
                weights = group_weights(groups[ids])
                offset = lower_weighted_median(y[ids], weights)
                kernel = kernels[feature, gamma]
                dual = np.linalg.solve(kernel[np.ix_(ids, ids)] + np.diag(alpha / weights), y[ids] - offset)
                cache[key] = np.clip(kernel[np.ix_(test, ids)] @ dual + offset, *bounds)
        return cache[key]
    return np.array([ridge('After', g, a, train, binary, (0, 1), 'positive_score') * ridge(f, g, a, positive, targets, (0, 1.25), 'amount') for f, g, a in PARAMETERS])

def forecast(records, kernels, targets, groups, episode):
    train = episode['train']
    positions = {i: j for j, i in enumerate(train)}
    inner = np.full((len(PARAMETERS), len(train)), np.nan)
    visits = np.zeros(len(train), dtype=int)
    check_partition(records, train, episode['test'])
    for fold in episode['inner']:
        if not set(fold['train'] + fold['validation']).issubset(positions):
            raise ValueError('Inner folds exceed the calibration pool.')
        check_partition(records, fold['train'], fold['validation'])
        ids = [positions[i] for i in fold['validation']]
        inner[:, ids] = predictions(kernels, targets, groups, fold['train'], fold['validation'])
        visits[ids] += 1
    if not np.all(visits == 1) or not np.isfinite(inner).all():
        raise ValueError('Invalid inner validation coverage.')
    scores = group_mae(inner, targets[train], groups[train])
    chosen = int(np.argmin(scores))
    result = predictions(kernels, targets, groups, train, episode['test'])[chosen]
    return (result, {'candidate': chosen, 'parameters': PARAMETERS[chosen], 'inner_mae_pp': float(scores[chosen] * 100)})

def run(output, root=ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_payload(root)
    protocol = (root / 'docs/HURDLE-PROTOCOL.json').read_bytes()
    expanded_bytes = (root / 'reports/data-budget.json').read_bytes()
    expanded = json.loads(expanded_bytes)
    records = json.loads((root / 'data/records.json').read_text())
    original = json.loads((root / 'data/episodes.json').read_text())
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    blocks = np.array([r['dependency_block'] for r in records])
    with np.load(root / 'data/features.npz', allow_pickle=False) as features:
        if features['record_ids'].tolist() != [r['record_id'] for r in records]:
            raise ValueError('Feature rows differ from labels.')
        representations = meal_representations(features['before'], features['after'])
    results = {'Hurdle50': np.full(len(records), np.nan), 'HurdleExpanded': np.full(len(records), np.nan)}
    selections = []
    started = time.monotonic()
    with threadpool_limits(limits=2):
        kernels = CandidateRegressors(representations, targets, groups).kernels
        for name, episodes in (('Hurdle50', original), ('HurdleExpanded', expanded['episodes'])):
            for episode in episodes:
                result, selection = forecast(records, kernels, targets, groups, episode)
                results[name][episode['test']] = result
                selections.append({'method': name, 'episode': episode['name'], **selection})
                print(f"Evaluated {name} {episode['name']}", flush=True)
    if not all((np.isfinite(x).all() for x in results.values())):
        raise ValueError('Missing outer forecasts.')
    baseline = np.array([r['predictions']['Change50'] for r in expanded['rows']])
    summary, comparison = ({}, {})
    for source in sorted({r['source'] for r in records}):
        ids = np.array([i for i, r in enumerate(records) if r['source'] == source])
        summary[source] = {name: summarize(targets[ids], p[ids], groups[ids]) for name, p in {'Change50': baseline, **results}.items()}
        comparison[source] = {}
        base = summary[source]['Change50']
        for name, p in results.items():
            new = summary[source][name]
            interval = paired_block_interval(targets[ids], p[ids], baseline[ids], groups[ids], blocks[ids])
            gates = {'MAE_reduction_at_least_10_percent': new['all']['group_mae_pp'] <= 0.9 * base['all']['group_mae_pp'], 'small_positive_degradation_at_most_1_pp': new['positive_up_to_10_percent']['group_mae_pp'] <= base['positive_up_to_10_percent']['group_mae_pp'] + 1, 'no_more_recorded_zero_predictions_above_10_percent': new['recorded_zero']['predictions_above_10_percent_count'] <= base['recorded_zero']['predictions_above_10_percent_count'], 'paired_interval_below_zero': interval['interval_95_pp'][1] < 0}
            comparison[source][name] = {**interval, 'gates': gates, 'all_gates_pass': all(gates.values())}
    report = {'protocol': json.loads(protocol), 'protocol_sha256': hashlib.sha256(protocol).hexdigest(), 'expanded_report_sha256': hashlib.sha256(expanded_bytes).hexdigest(), 'elapsed_seconds': time.monotonic() - started, 'summary': summary, 'comparison': comparison, 'selections': selections, 'rows': [{**r, 'predictions': {name: float(p[i]) for name, p in results.items()}} for i, r in enumerate(records)]}
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report
if __name__ == '__main__':
    import sys
    report = run(sys.argv[1])
    print(json.dumps({s: {k: v['all']['group_mae_pp'] for k, v in x.items()} for s, x in report['summary'].items()}, indent=2))
