"""Reproduce held-out errors without changing labels, splits or the demo model."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import ROOT, verify_payload
from .features import meal_representations
from .inference import Predictor
from .models import CandidateRegressors, PREDICTION_BOUNDS, candidate_banks, group_weights
from .selection import check_partition, evaluate_episode, group_mae
SLICES = ('all', 'recorded_zero', 'positive_up_to_10_percent', 'above_10_percent')
METHODS = ('Mean', 'Median', 'Appearance', 'Change', 'After', 'Paired', 'Vector change', 'Cosine change')

def summarize(targets, predictions, groups):
    """Report both group-balanced errors and explicitly row-weighted tail counts."""
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    groups = np.asarray(groups)
    if predictions.shape != targets.shape or groups.shape != targets.shape:
        raise ValueError('Targets, predictions and groups must be aligned vectors.')
    if targets.ndim != 1 or not np.isfinite(targets).all() or (not np.isfinite(predictions).all()):
        raise ValueError('Diagnostic inputs must be finite vectors.')
    predictions = np.clip(predictions, *PREDICTION_BOUNDS)
    masks = (np.ones(len(targets), dtype=bool), targets == 0, (targets > 0) & (targets <= 0.1), targets > 0.1)
    summary = {}
    for name, mask in zip(SLICES, masks):
        y, p, g = (targets[mask], predictions[mask], groups[mask])
        errors = np.abs(p - y)
        summary[name] = {'records': len(y), 'groups': len(set(g)), 'group_mae_pp': float(100 * group_mae(p[None, :], y, g)[0]) if len(y) else None, 'row_mae_pp': float(100 * errors.mean()) if len(y) else None, 'row_p90_absolute_error_pp': float(100 * np.quantile(errors, 0.9)) if len(y) else None, 'row_max_absolute_error_pp': float(100 * errors.max()) if len(y) else None, 'predictions_above_10_percent_count': int(np.sum(p > 0.1)), 'predictions_at_or_below_1_percent_count': int(np.sum(p <= 0.01))}
    return summary

def run(output: Path, root: Path=ROOT) -> dict:
    """Refit existing banks, verify reference forecasts, then report failure slices.

    Every representation uses its original inner-selected candidate. Outer labels
    only score forecasts. Examining these errors makes this a development audit,
    not a new final test or a license to select a replacement on these scores.
    """
    if output.exists():
        raise FileExistsError(f'Choose a new output path: {output}')
    verify_payload(root)
    records = json.loads((root / 'data/records.json').read_text())
    episodes = json.loads((root / 'data/episodes.json').read_text())
    references = json.loads((root / 'data/reference_predictions.json').read_text())
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    with np.load(root / 'data/features.npz', allow_pickle=False) as data:
        if data['record_ids'].tolist() != [r['record_id'] for r in records]:
            raise ValueError('Feature and label row identifiers differ.')
        before, after = (data['before'].copy(), data['after'].copy())
    representations = meal_representations(before, after)
    predictions = np.full((len(METHODS), len(records)), np.nan)
    visits = np.zeros(len(records), dtype=int)
    selections, episode_names = ([], [None] * len(records))
    with threadpool_limits(limits=2):
        regressors = CandidateRegressors(representations, targets, groups)
        for episode in episodes:
            result = evaluate_episode(records, episode, regressors)
            selected = dict(zip(candidate_banks(), map(int, result.selected_candidates)))
            baseline_ids = [0, selected['Appearance'], selected['Change']]
            reference = references[episode['name']]
            if baseline_ids != reference['candidate_ids']:
                raise ValueError('Baseline candidate differs from the reference.')
            np.testing.assert_allclose(result.test_forecasts.predictions[baseline_ids], reference['raw_predictions'], rtol=0, atol=1e-10)
            train, test = (episode['train'], episode['test'])
            predictions[0, test] = np.average(targets[train], weights=group_weights(groups[train]))
            for j, method in enumerate(METHODS[1:], start=1):
                candidate = 0 if method == 'Median' else selected[method]
                predictions[j, test] = result.test_forecasts.predictions[candidate]
            visits[test] += 1
            for i in test:
                episode_names[i] = episode['name']
            selections.append({'episode': episode['name'], 'candidates': selected})
            print(f"Diagnosed {episode['name']}", flush=True)
    if not np.all(visits == 1) or not np.isfinite(predictions).all():
        raise ValueError('Each record must have exactly one finite outer forecast.')
    predictions = np.clip(predictions, *PREDICTION_BOUNDS)
    summary = {}
    for source in sorted({r['source'] for r in records}):
        ids = [i for i, r in enumerate(records) if r['source'] == source]
        summary[source] = {name: summarize(targets[ids], predictions[j, ids], groups[ids]) for j, name in enumerate(METHODS)}
    predictor = Predictor(root)
    demo_episode = next((e for e in episodes if e['name'] == predictor.metadata['episode']))
    if [records[i]['record_id'] for i in demo_episode['train']] != predictor.metadata['train_ids']:
        raise ValueError('Demo calibration IDs differ from its declared episode.')
    ids = demo_episode['test']
    check_partition(records, demo_episode['train'], ids)
    demo_predictions = predictor.predict_embeddings(before[ids], after[ids])
    np.testing.assert_allclose(demo_predictions, predictions[METHODS.index('Change'), ids], rtol=0, atol=1e-10)
    demo_rows = [{'record_id': records[i]['record_id'], 'group': records[i]['group'], 'recorded_fraction': float(targets[i]), 'estimated_fraction': float(p), 'absolute_error_pp': float(100 * abs(p - targets[i]))} for i, p in zip(ids, demo_predictions)]
    report = {'scope': 'Development diagnostic on existing outer folds; not an untouched final test. No labels were corrected or records excluded.', 'zero_label_caveat': 'Recorded zero mass does not establish visually empty food regions. Source image/weight inconsistencies require separate review.', 'units': 'Fractions are 0–1.25; errors ending in pp are percentage points. Tail counts and quantiles count rows, not groups.', 'selection': 'Original nested grouped selection and calibration budgets. Per-feature banks have 12 candidates; Appearance and Change each have 25 including Median.', 'input_manifest_sha256': hashlib.sha256((root / 'data/checksums.json').read_bytes()).hexdigest(), 'summary': summary, 'selections': selections, 'demo': {'episode': demo_episode['name'], 'summary': summarize(targets[ids], demo_predictions, groups[ids]), 'rows': sorted(demo_rows, key=lambda r: (-r['absolute_error_pp'], r['record_id']))}, 'rows': [{**record, 'episode': episode_names[i], 'predictions': {name: float(predictions[j, i]) for j, name in enumerate(METHODS)}} for i, record in enumerate(records)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
