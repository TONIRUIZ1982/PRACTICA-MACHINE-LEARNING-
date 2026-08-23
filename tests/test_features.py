import unittest

import pandas as pd

from src.hotel_cancellation import EXPECTED_COLUMNS
from src.hotel_cancellation.features import add_engineered_features, prepare_model_frame


def booking_row(**overrides) -> dict:
    row = {column: 0 for column in EXPECTED_COLUMNS}
    row.update(
        {
            "hotel": "City Hotel",
            "arrival_date_year": 2016,
            "arrival_date_month": "January",
            "arrival_date_week_number": 1,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "reservation_status": "Check-Out",
            "reservation_status_date": "2016-01-02",
        }
    )
    row.update(overrides)
    return row


class FeatureEngineeringTests(unittest.TestCase):
    def test_interpretable_totals_are_created(self):
        dataframe = pd.DataFrame(
            [booking_row(stays_in_weekend_nights=2, stays_in_week_nights=3, adults=2, children=1)]
        )
        prepared = add_engineered_features(dataframe)
        self.assertEqual(prepared.loc[0, "total_nights"], 5)
        self.assertEqual(prepared.loc[0, "total_guests"], 3)
        self.assertEqual(prepared.loc[0, "has_children"], 1)

    def test_model_frame_excludes_leakage_and_operational_columns(self):
        dataframe = pd.DataFrame([booking_row()])
        predictors, target = prepare_model_frame(dataframe)
        for excluded in (
            "is_canceled",
            "reservation_status",
            "reservation_status_date",
            "assigned_room_type",
            "booking_changes",
            "days_in_waiting_list",
            "company",
            "arrival_date_year",
        ):
            self.assertNotIn(excluded, predictors.columns)
        self.assertEqual(target.name, "is_canceled")


if __name__ == "__main__":
    unittest.main()
