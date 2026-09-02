import unittest

import pandas as pd

from src.hotel_cancellation.comparison import select_best_model
from src.hotel_cancellation.model_suite import MODEL_ORDER, build_required_models


class RequiredModelTests(unittest.TestCase):
    def setUp(self):
        self.predictors = pd.DataFrame(
            {
                "lead_time": [5.0, 20.0, 60.0, 90.0],
                "hotel": ["City", "Resort", "City", "Resort"],
            }
        )

    def test_factory_contains_every_mandatory_model(self):
        models = build_required_models(self.predictors, neural_epochs=1)
        self.assertEqual(tuple(models), MODEL_ORDER)
        for model in models.values():
            self.assertEqual(tuple(model.named_steps), ("preprocessing", "classifier"))

    def test_best_model_uses_f1_by_default(self):
        results = {
            "model_a": {"f1": 0.60, "roc_auc": 0.90},
            "model_b": {"f1": 0.70, "roc_auc": 0.80},
        }
        self.assertEqual(select_best_model(results), "model_b")

    def test_missing_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            select_best_model({"model": {"roc_auc": 0.80}})


if __name__ == "__main__":
    unittest.main()
