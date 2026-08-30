"""Compare the fixed 50-label baseline with all eligible calibration records."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import ROOT, verify_payload
from .diagnostics import summarize
from .features import meal_representations
from .models import CandidateRegressors
from .selection import check_partition, evaluate_episode
from .segmentation_experiment import paired_block_interval

def eligible(records, candidates, held_out, hashes):
    forbidden = {key: {records[i][key] for i in held_out} for key in ('record_id', 'group', 'similarity_component')}
    forbidden_images = set().union(*(hashes.get(records[i]['record_id'], set()) for i in held_out))
    return [i for i in candidates if all((records[i][key] not in values for key, values in forbidden.items())) and (not hashes.get(records[i]['record_id'], set()) & forbidden_images)]

def expanded_episode(records, original, hashes):
    pool = [i for i, row in enumerate(records) if row['source'] == original['source']]
    train = eligible(records, pool, original['test'], hashes)
    buckets = [[] for _ in range(5)]
    groups = sorted({records[i]['group'] for i in train})
    members = {g: [i for i in train if records[i]['group'] == g] for g in groups}
    for group in sorted(groups, key=lambda g: (-len(members[g]), g)):
        min(buckets, key=len).extend(members[group])
    inner = []
    for validation in buckets:
        if not validation:
            continue
        fitting = eligible(records, train, validation, hashes)
        if not fitting:
            raise ValueError('Expanded inner split has no eligible training records.')
        check_partition(records, fitting, validation)
        inner.append({'train': fitting, 'validation': validation})
    check_partition(records, train, original['test'])
    return {**original, 'name': original['name'] + '_expanded', 'budget': len(train), 'train': train, 'inner': inner}

def run(output, root=ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    verify_payload(root)
    protocol = (root / 'docs/DATA-BUDGET-PROTOCOL.json').read_bytes()
    records = json.loads((root / 'data/records.json').read_text())
    episodes = json.loads((root / 'data/episodes.json').read_text())
    refs = json.loads((root / 'data/reference_predictions.json').read_text())
    queue = json.loads((root / 'reports/review-queue/review-queue.json').read_text())
    hashes = {r['record_id']: {r['images'][k]['sha256'] for k in ('before', 'after')} for r in queue['rows']}
    with np.load(root / 'data/features.npz', allow_pickle=False) as features:
        if features['record_ids'].tolist() != [r['record_id'] for r in records]:
            raise ValueError('Feature rows do not match records.')
        representations = meal_representations(features['before'], features['after'])
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    blocks = np.array([r['dependency_block'] for r in records])
    predictions = {name: np.full(len(records), np.nan) for name in ('Change50', 'Appearance50', 'ChangeExpanded', 'AppearanceExpanded')}
    expanded = [expanded_episode(records, e, hashes) for e in episodes]
    selections = []
    started = time.monotonic()
    with threadpool_limits(limits=2):
        regressors = CandidateRegressors(representations, targets, groups)
        for original, episode in zip(episodes, expanded):
            result = evaluate_episode(records, episode, regressors)
            selections.append({'episode': episode['name'], 'training_pairs': episode['budget'], 'appearance_candidate': int(result.selected_candidates[0]), 'change_candidate': int(result.selected_candidates[1]), 'inner_train_sizes': [len(f['train']) for f in episode['inner']]})
            for index, family in enumerate(('Appearance', 'Change')):
                predictions[family + '50'][episode['test']] = np.clip(refs[original['name']]['raw_predictions'][index + 1], 0, 1.25)
                predictions[family + 'Expanded'][episode['test']] = np.clip(result.test_forecasts.predictions[result.selected_candidates[index]], 0, 1.25)
            print(f"Evaluated {episode['name']} with {episode['budget']} training pairs", flush=True)
    if not all((np.isfinite(p).all() for p in predictions.values())):
        raise ValueError('Missing or invalid outer forecasts.')
    summary, comparison = ({}, {})
    for source in sorted({r['source'] for r in records}):
        ids = np.array([i for i, r in enumerate(records) if r['source'] == source])
        summary[source] = {name: summarize(targets[ids], p[ids], groups[ids]) for name, p in predictions.items()}
        base, new = (summary[source]['Change50'], summary[source]['ChangeExpanded'])
        interval = paired_block_interval(targets[ids], predictions['ChangeExpanded'][ids], predictions['Change50'][ids], groups[ids], blocks[ids])
        gates = {'MAE_reduction_at_least_10_percent': new['all']['group_mae_pp'] <= 0.9 * base['all']['group_mae_pp'], 'small_positive_degradation_at_most_1_pp': new['positive_up_to_10_percent']['group_mae_pp'] <= base['positive_up_to_10_percent']['group_mae_pp'] + 1, 'no_more_recorded_zero_predictions_above_10_percent': new['recorded_zero']['predictions_above_10_percent_count'] <= base['recorded_zero']['predictions_above_10_percent_count'], 'paired_interval_below_zero': interval['interval_95_pp'][1] < 0}
        comparison[source] = {**interval, 'gates': gates, 'all_gates_pass': all(gates.values())}
    report = {'protocol': json.loads(protocol), 'protocol_sha256': hashlib.sha256(protocol).hexdigest(), 'elapsed_seconds': time.monotonic() - started, 'summary': summary, 'comparison': comparison, 'selections': selections, 'episodes': expanded, 'rows': [{**r, 'predictions': {name: float(p[i]) for name, p in predictions.items()}} for i, r in enumerate(records)]}
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report
if __name__ == '__main__':
    import sys
    report = run(sys.argv[1])
    print(json.dumps({s: {k: v['all']['group_mae_pp'] for k, v in x.items()} for s, x in report['summary'].items()}, indent=2))
