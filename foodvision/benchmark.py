"""Recompute a compact grouped benchmark from the bundled DINOv2 features."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent.parent
METHODS = ('Mean', 'Median', 'Appearance', 'Change')

def verify_payload(root: Path=ROOT) -> int:
    manifest = json.loads((root / 'data/checksums.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundled input is missing or changed: {name}')
    return len(manifest)

def check_partition(records: list[dict], train: list[int], held_out: list[int]) -> None:
    """Catch crossed record, original-group and reviewed-image boundaries."""
    for field in ('record_id', 'group', 'similarity_component'):
        training_values = {records[index][field] for index in train}
        held_out_values = {records[index][field] for index in held_out}
        if training_values & held_out_values:
            raise ValueError(f'Training and evaluation overlap in {field}.')
from .models import group_weights, lower_weighted_median

def run(output, root=ROOT):
    if output.exists():
        raise FileExistsError(f'Choose a new output path: {output}')
    verify_payload(root)
    records = json.loads((root / 'data/records.json').read_text())
    episodes = json.loads((root / 'data/episodes.json').read_text())
    targets = np.array([r['fraction'] for r in records])
    groups = np.array([r['group'] for r in records])
    predictions = np.full((2, len(records)), np.nan)
    for episode in episodes:
        train, test = (episode['train'], episode['test'])
        check_partition(records, train, test)
        weights = group_weights(groups[train])
        predictions[0, test] = np.average(targets[train], weights=weights)
        predictions[1, test] = lower_weighted_median(targets[train], weights)
    if not np.isfinite(predictions).all():
        raise ValueError('Missing held-out prediction.')
    summary = {}
    for source in ('LeFood', 'ACETADA'):
        ids = np.array([i for i, r in enumerate(records) if r['source'] == source])
        scores = group_mae(predictions[:, ids], targets[ids], groups[ids])
        summary[source] = {'records': len(ids), 'mae_percentage_points': dict(zip(('Mean', 'Median'), (100 * scores).tolist()))}
    report = {'summary': summary, 'episodes': len(episodes)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report

def group_mae(predictions, fractions, groups):
    errors = np.abs(np.clip(predictions, 0, 1.25) - fractions)
    return np.mean([errors[:, groups == group].mean(axis=1) for group in sorted(set(groups))], axis=0)
