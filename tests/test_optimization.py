import unittest

import numpy as np
import pandas as pd

from src.hotel_cancellation.optimization import (
    best_f1_threshold,
    build_classical_searches,
    build_neural_candidates,
)


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.predictors = pd.DataFrame(
            {
                "lead_time": [5.0, 20.0, 60.0, 90.0, 10.0, 80.0],
                "hotel": ["City", "Resort", "City", "Resort", "City", "Resort"],
            }
        )
        self.target = pd.Series([0, 0, 1, 1, 0, 1])

    def test_threshold_maximizes_f1(self):
        probabilities = np.array([0.1, 0.3, 0.4, 0.8])
        target = np.array([0, 1, 1, 1])
        threshold, score = best_f1_threshold(target, probabilities)
        self.assertAlmostEqual(threshold, 0.3)
        self.assertAlmostEqual(score, 1.0)

    def test_classical_searches_cover_four_models(self):
        searches = build_classical_searches(
            self.predictors,
            self.target,
            iterations=1,
            cv_splits=2,
            n_jobs=1,
        )
        self.assertEqual(
            set(searches),
            {"logistic_regression", "decision_tree", "random_forest", "xgboost"},
        )

    def test_three_neural_architectures_are_available(self):
        candidates = build_neural_candidates(self.predictors, epochs=1)
        self.assertEqual(len(candidates), 3)


if __name__ == "__main__":
    unittest.main()
