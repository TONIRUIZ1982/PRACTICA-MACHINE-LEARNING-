import unittest

import pandas as pd

from src.hotel_cancellation.modeling import build_logistic_pipeline, evaluate_classifier


class ModelingTests(unittest.TestCase):
    def test_pipeline_handles_missing_and_unseen_categories(self):
        train_x = pd.DataFrame(
            {
                "lead_time": [5.0, None, 60.0, 90.0, 10.0, 80.0],
                "hotel": ["City", "Resort", "City", "Resort", "City", "Resort"],
            }
        )
        train_y = pd.Series([0, 0, 1, 1, 0, 1])
        validation_x = pd.DataFrame(
            {"lead_time": [None, 45.0], "hotel": ["New hotel", "City"]}
        )
        validation_y = pd.Series([0, 1])

        model = build_logistic_pipeline(train_x)
        model.fit(train_x, train_y)
        metrics = evaluate_classifier(model, validation_x, validation_y)

        self.assertEqual(metrics["samples"], 2)
        self.assertTrue(0 <= metrics["f1"] <= 1)
        self.assertTrue(0 <= metrics["roc_auc"] <= 1)


if __name__ == "__main__":
    unittest.main()
