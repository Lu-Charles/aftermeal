"""Represent a meal pair using the saved DINOv2 image embeddings."""
import numpy as np

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize each image independently; no statistics come from test images."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / lengths

def meal_representations(before: np.ndarray, after: np.ndarray) -> dict[str, np.ndarray]:
    """Build the four representations used in the matched comparison.

    Dividing concatenated features by sqrt(2) keeps paired-vector lengths
    comparable to single-image lengths. Change magnitude is retained: a small
    difference and a large difference should not become the same unit vector.
    """
    before = normalize_embeddings(before)
    after = normalize_embeddings(after)
    return {'After': after, 'Paired': np.concatenate([before, after], axis=1) / np.sqrt(2), 'Vector change': (after - before) / np.sqrt(2), 'Cosine change': (1 - (after * before).sum(axis=1))[:, None]}

def rbf_kernel(features: np.ndarray, gamma: float) -> np.ndarray:
    """Cache pairwise similarities without using any food-mass labels."""
    squared_lengths = (features * features).sum(axis=1)
    distances = squared_lengths[:, None] + squared_lengths[None, :] - 2 * features @ features.T
    return np.exp(-gamma * np.maximum(distances, 0))
