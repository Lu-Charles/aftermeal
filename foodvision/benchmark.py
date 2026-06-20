"""Recompute a compact grouped benchmark from the bundled DINOv2 features."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .features import meal_representations
from .models import CandidateRegressors, group_weights
from .selection import evaluate_episode
ROOT = Path(__file__).resolve().parent.parent
METHODS = ('Mean', 'Median', 'Appearance', 'Change')

def verify_payload(root: Path=ROOT) -> int:
    manifest = json.loads((root / 'data/checksums.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundled input is missing or changed: {name}')
    return len(manifest)

def run(output: Path, root: Path=ROOT) -> dict:
    if output.exists():
        raise FileExistsError(f'Choose a new output path: {output}')
    started = time.monotonic()
    verify_payload(root)
    records = json.loads((root / 'data/records.json').read_text())
    episodes = json.loads((root / 'data/episodes.json').read_text())
    with np.load(root / 'data/features.npz', allow_pickle=False) as features:
        if features['record_ids'].tolist() != [r['record_id'] for r in records]:
            raise ValueError('Feature and label row identifiers differ.')
        representations = meal_representations(features['before'], features['after'])
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    predictions = np.full((len(METHODS), len(records)), np.nan)
    visits = np.zeros(len(records), dtype=int)
    selections = []
    maximum_difference = 0.0
    with threadpool_limits(limits=2):
        regressors = CandidateRegressors(representations, targets, groups)
        for episode in episodes:
            result = evaluate_episode(records, episode, regressors)
            ids = [0, int(result.selected_candidates[0]), int(result.selected_candidates[1])]
            raw = result.test_forecasts.predictions[ids]
            train, test = (episode['train'], episode['test'])
            predictions[0, test] = np.average(targets[train], weights=group_weights(groups[train]))
            predictions[1:, test] = raw
            visits[test] += 1
            selections.append({'episode': episode['name'], 'appearance_candidate': ids[1], 'change_candidate': ids[2]})
            print(f"Checked {episode['name']}", flush=True)
    if not np.all(visits == 1) or not np.isfinite(predictions).all():
        raise ValueError('Every record must receive one finite held-out forecast per method.')
    errors = np.abs(np.clip(predictions, 0, 1.25) - targets)
    summary = {}
    for source in ('LeFood', 'ACETADA'):
        indices = np.array([i for i, r in enumerate(records) if r['source'] == source])
        source_groups = sorted(set(groups[indices]))
        macro = np.mean([errors[:, indices[groups[indices] == g]].mean(axis=1) for g in source_groups], axis=0)
        summary[source] = {'records': len(indices), 'groups': len(source_groups), 'mae_percentage_points': dict(zip(METHODS, (100 * macro).tolist()))}
    report = {'design': 'Five outer folds per source, one fixed calibration repetition, 50 labels per fit; nested grouped selection with similarity purges.', 'scope': 'Compact feature-cache benchmark. It does not retrain the image encoder or reproduce the separate CNN study.', 'episodes': len(episodes), 'records': len(records), 'label_budget': 50, 'summary': summary, 'selections': selections, 'maximum_reference_prediction_difference': maximum_difference, 'elapsed_seconds': round(time.monotonic() - started, 3)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
