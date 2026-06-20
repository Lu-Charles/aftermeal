"""Choose candidates using held-out calibration groups, never outer-test labels."""
from dataclasses import dataclass
import numpy as np
from .models import CANDIDATES, PREDICTION_BOUNDS, CandidateRegressors, Forecasts, candidate_banks

def check_partition(records: list[dict], train: list[int], held_out: list[int]) -> None:
    """Catch crossed record, original-group and reviewed-image boundaries."""
    for field in ('record_id', 'group', 'similarity_component'):
        training_values = {records[index][field] for index in train}
        held_out_values = {records[index][field] for index in held_out}
        if training_values & held_out_values:
            raise ValueError(f'Training and evaluation overlap in {field}.')

def group_mae(predictions, fractions, groups) -> np.ndarray:
    """Average within groups first, so large groups do not dominate selection."""
    bounded = np.clip(predictions, *PREDICTION_BOUNDS)
    errors = np.abs(bounded - fractions)
    return np.mean([errors[:, groups == group].mean(axis=1) for group in sorted(set(groups))], axis=0)

@dataclass
class EpisodeResult:
    inner_predictions: np.ndarray
    validation_scores: np.ndarray
    selected_candidates: np.ndarray
    test_forecasts: Forecasts

    def save(self, path) -> None:
        """Retain the original array names for comparison with archived runs."""
        np.savez_compressed(path, cv_predictions=self.inner_predictions, cv_scores=self.validation_scores, chosen=self.selected_candidates, raw_predictions=self.test_forecasts.predictions, duals=self.test_forecasts.coefficients, offset=self.test_forecasts.offset)

def evaluate_episode(records, episode, regressors: CandidateRegressors) -> EpisodeResult:
    calibration_ids = episode['train']
    test_ids = episode['test']
    if len(calibration_ids) != episode['budget'] or len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError('Calibration IDs do not match the declared label budget.')
    for partition in episode['inner']:
        check_partition(records, partition['train'], partition['validation'])
    check_partition(records, calibration_ids, test_ids)
    calibration_positions = {record_id: index for index, record_id in enumerate(calibration_ids)}
    inner_predictions = np.full((len(CANDIDATES), len(calibration_ids)), np.nan)
    validation_visits = []
    for partition in episode['inner']:
        training_ids = partition['train']
        validation_ids = partition['validation']
        if not set(training_ids + validation_ids).issubset(calibration_positions):
            raise ValueError('An inner fold uses records outside its calibration budget.')
        predictions = regressors.predict(training_ids, validation_ids).predictions
        positions = [calibration_positions[record_id] for record_id in validation_ids]
        inner_predictions[:, positions] = predictions
        validation_visits.extend(validation_ids)
    if sorted(validation_visits) != sorted(calibration_ids):
        raise ValueError('Each calibration record must receive exactly one validation forecast.')
    if not np.isfinite(inner_predictions).all():
        raise ValueError('Inner validation produced missing or non-finite forecasts.')
    validation_scores = group_mae(inner_predictions, regressors.fractions[calibration_ids], regressors.groups[calibration_ids])
    selected = [indices[int(np.argmin(validation_scores[indices]))] for indices in candidate_banks().values()]
    return EpisodeResult(inner_predictions, validation_scores, np.array(selected), regressors.predict(calibration_ids, test_ids))
