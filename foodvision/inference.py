"""Load the compact regression head and encode new meal photographs."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .features import meal_representations
ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20000000
ENCODER_LOCK = threading.Lock()

class Predictor:

    def __init__(self, root: Path=ROOT):
        self.root = root
        self.metadata = json.loads((root / 'models/demo_model.json').read_text())
        with np.load(root / 'models/demo_model.npz', allow_pickle=False) as model:
            self.training = model['train_features'].copy()
            self.duals = model['duals'].copy()
            self.offset = float(model['offset'])

    def predict_embeddings(self, before, after) -> np.ndarray:
        features = meal_representations(before, after)[self.metadata['feature']]
        if features.shape[1] != self.training.shape[1]:
            raise ValueError('Image features do not match this model.')
        distance = np.maximum((features * features).sum(axis=1)[:, None] + (self.training * self.training).sum(axis=1)[None, :] - 2 * features @ self.training.T, 0)
        prediction = np.exp(-self.metadata['gamma'] * distance) @ self.duals + self.offset
        return np.clip(prediction, *self.metadata['bounds'])

    def example(self, record_id: str) -> dict:
        examples = json.loads((self.root / 'examples/examples.json').read_text())
        index = next((i for i, item in enumerate(examples) if item['id'] == record_id), None)
        if index is None:
            raise ValueError('Unknown example.')
        with np.load(self.root / 'examples/features.npz', allow_pickle=False) as features:
            fraction = self.predict_embeddings(features['before'][[index]], features['after'][[index]])[0]
        return {**examples[index], 'estimated_fraction': float(fraction), 'input_mode': 'saved image embeddings'}
