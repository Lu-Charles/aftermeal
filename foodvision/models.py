"""Group-balanced constant baselines."""
from collections import Counter
import numpy as np

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
