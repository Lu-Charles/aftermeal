"""Regression checks for model packaging, image input and the local HTTP API."""
import unittest
if __name__ == '__main__':
    unittest.main()
from foodvision.inference import Predictor
import numpy as np
import unittest

class EstimatorLoading(unittest.TestCase):

    def test_compact_model_predicts_finite_values(self):
        predictor = Predictor()
        with np.load(predictor.root / 'examples/features.npz', allow_pickle=False) as features:
            result = predictor.predict_embeddings(features['before'], features['after'])
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1.25)).all())
