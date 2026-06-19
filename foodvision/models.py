"""The fixed candidate models and their group-weighted regression fits."""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .features import rbf_kernel
FEATURE_NAMES = ('After', 'Paired', 'Vector change', 'Cosine change')
GAMMAS = (0.5, 2, 8)
PENALTIES = (0.01, 0.1, 1, 10)
PREDICTION_BOUNDS = (0, 1.25)

@dataclass(frozen=True)
class Candidate:
    feature: str
    gamma: float | None = None
    alpha: float | None = None
CANDIDATES = (Candidate('Median'),) + tuple((Candidate(feature, gamma, penalty) for feature in FEATURE_NAMES for gamma in GAMMAS for penalty in PENALTIES))

def candidate_banks() -> dict[str, list[int]]:
    """Give appearance and change models equal-sized parameter searches."""
    feature_ids = {name: [index for index, model in enumerate(CANDIDATES) if model.feature == name] for name in FEATURE_NAMES}
    return {'Appearance': [0] + feature_ids['After'] + feature_ids['Paired'], 'Change': [0] + feature_ids['Vector change'] + feature_ids['Cosine change'], 'Expanded': list(range(len(CANDIDATES))), **feature_ids}

def group_weights(groups: np.ndarray) -> np.ndarray:
    """Give each observed food/participant group equal total fitting weight."""
    counts = Counter(groups)
    weights = np.array([1 / counts[group] for group in groups])
    if not len(weights):
        raise ValueError('Cannot weight an empty training set.')
    return weights / weights.mean()

def lower_weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Use the lower endpoint when exactly half the weight lies on each side.

    The tolerance handles accumulated floating-point error at a 50/50 tie.
    This convention was fixed during verification of the original experiments.
    """
    order = np.argsort(values, kind='stable')
    threshold = weights.sum() / 2 - 1e-12 * weights.sum()
    position = np.searchsorted(np.cumsum(weights[order]), threshold)
    return float(values[order[position]])

@dataclass
class Forecasts:
    predictions: np.ndarray
    coefficients: np.ndarray
    offset: float

class CandidateRegressors:
    """Fit the same candidate list on each calibration or inner-training set."""

    def __init__(self, representations, fractions, groups):
        self.fractions = np.asarray(fractions, dtype=float)
        self.groups = np.asarray(groups)
        self.kernels = {(name, gamma): rbf_kernel(features, gamma) for name, features in representations.items() for gamma in GAMMAS}

    def predict(self, train_ids: list[int], test_ids: list[int]) -> Forecasts:
        if not train_ids:
            return Forecasts(predictions=np.zeros((len(CANDIDATES), len(test_ids))), coefficients=np.zeros((len(CANDIDATES) - 1, 0)), offset=0.0)
        train = np.asarray(train_ids, dtype=int)
        test = np.asarray(test_ids, dtype=int)
        weights = group_weights(self.groups[train])
        offset = lower_weighted_median(self.fractions[train], weights)
        centered_targets = self.fractions[train] - offset
        predictions = [np.full(len(test), offset)]
        coefficients = []
        for candidate in CANDIDATES[1:]:
            kernel = self.kernels[candidate.feature, candidate.gamma]
            training_kernel = kernel[np.ix_(train, train)]
            system = training_kernel + np.diag(candidate.alpha / weights)
            fitted_coefficients = np.linalg.solve(system, centered_targets)
            predictions.append(kernel[np.ix_(test, train)] @ fitted_coefficients + offset)
            coefficients.append(fitted_coefficients)
        return Forecasts(np.array(predictions), np.array(coefficients), offset)
