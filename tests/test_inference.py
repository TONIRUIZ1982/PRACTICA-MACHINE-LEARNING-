import unittest

import numpy as np
import pandas as pd

from src.hotel_cancellation import DatasetValidationError, INFERENCE_INPUT_COLUMNS
from src.hotel_cancellation.inference import predict_cancellations
from src.hotel_cancellation.features import prepare_inference_frame


def reservation_row(**overrides) -> dict:
    row = {
        "hotel": "City Hotel",
        "lead_time": 40,
        "arrival_date_week_number": 20,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
    }
    row.update(overrides)
    return row


class FakeProbabilityModel:
    def predict_proba(self, predictors):
        return np.tile(np.array([0.25, 0.75]), (len(predictors), 1))


class InferenceTests(unittest.TestCase):
    def test_inference_frame_matches_trained_predictor_contract(self):
        predictors = prepare_inference_frame(pd.DataFrame([reservation_row()]))
        self.assertEqual(predictors.shape, (1, 23))
        self.assertIn("has_children", predictors)
        self.assertIn("arrival_week_sin", predictors)

    def test_missing_input_is_rejected(self):
        incomplete = reservation_row()
        incomplete.pop(INFERENCE_INPUT_COLUMNS[0])
        with self.assertRaisesRegex(DatasetValidationError, "Missing inference"):
            prepare_inference_frame(pd.DataFrame([incomplete]))

    def test_prediction_returns_probability_and_decision(self):
        predictions = predict_cancellations(
            FakeProbabilityModel(),
            pd.DataFrame([reservation_row(), reservation_row()]),
            threshold=0.5,
        )
        self.assertEqual(predictions["predicted_is_canceled"].tolist(), [1, 1])
        self.assertTrue((predictions["cancellation_probability"] == 0.75).all())


if __name__ == "__main__":
    unittest.main()
