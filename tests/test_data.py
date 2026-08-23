from pathlib import Path
import unittest

import pandas as pd

from src.hotel_cancellation import (
    DatasetValidationError,
    EXPECTED_COLUMNS,
    LEAKAGE_FEATURES,
    TARGET_COLUMN,
    TIME_DEPENDENT_FEATURES,
    build_arrival_date,
    get_candidate_features,
    load_dataset,
    remove_exact_duplicates,
    split_by_arrival_date,
    validate_dataset,
)


LOCAL_DATASET = Path("data/raw/dataset_practica_final.csv")


def booking_row(**overrides) -> dict:
    row = {column: 0 for column in EXPECTED_COLUMNS}
    row.update(
        {
            "hotel": "City Hotel",
            "is_canceled": 0,
            "arrival_date_year": 2016,
            "arrival_date_month": "January",
            "arrival_date_day_of_month": 1,
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


class DataValidationTests(unittest.TestCase):
    def test_valid_schema_and_binary_target(self):
        dataframe = pd.DataFrame([booking_row(), booking_row(is_canceled=1)])
        validate_dataset(dataframe)

    def test_missing_column_is_rejected(self):
        dataframe = pd.DataFrame([booking_row()]).drop(columns=["hotel"])
        with self.assertRaisesRegex(DatasetValidationError, "Missing"):
            validate_dataset(dataframe)

    def test_non_binary_target_is_rejected(self):
        dataframe = pd.DataFrame([booking_row(is_canceled=2)])
        with self.assertRaisesRegex(DatasetValidationError, "binary"):
            validate_dataset(dataframe)

    def test_arrival_date_does_not_use_status_date(self):
        dataframe = pd.DataFrame(
            [
                booking_row(
                    arrival_date_year=2017,
                    arrival_date_month="August",
                    arrival_date_day_of_month=31,
                    reservation_status_date="2015-01-01",
                )
            ]
        )
        arrival = build_arrival_date(dataframe)
        self.assertEqual(arrival.iloc[0], pd.Timestamp("2017-08-31"))

    def test_unknown_month_is_rejected(self):
        dataframe = pd.DataFrame([booking_row(arrival_date_month="Unknown")])
        with self.assertRaisesRegex(DatasetValidationError, "months"):
            build_arrival_date(dataframe)


class PreparationTests(unittest.TestCase):
    def test_exact_duplicates_are_counted(self):
        dataframe = pd.DataFrame([booking_row(), booking_row()])
        deduplicated, removed = remove_exact_duplicates(dataframe)
        self.assertEqual(removed, 1)
        self.assertEqual(len(deduplicated), 1)

    def test_candidate_features_always_exclude_leakage(self):
        dataframe = pd.DataFrame([booking_row()])
        candidates = get_candidate_features(dataframe)
        self.assertNotIn(TARGET_COLUMN, candidates)
        self.assertTrue(set(LEAKAGE_FEATURES).isdisjoint(candidates))
        self.assertTrue(set(TIME_DEPENDENT_FEATURES).isdisjoint(candidates))
        self.assertNotIn("company", candidates)

    def test_temporal_split_is_non_overlapping(self):
        dataframe = pd.DataFrame(
            [
                booking_row(
                    arrival_date_year=2016,
                    arrival_date_month="September",
                    arrival_date_day_of_month=30,
                ),
                booking_row(
                    arrival_date_year=2016,
                    arrival_date_month="October",
                    arrival_date_day_of_month=1,
                ),
                booking_row(
                    arrival_date_year=2017,
                    arrival_date_month="January",
                    arrival_date_day_of_month=1,
                ),
            ]
        )
        temporal = split_by_arrival_date(dataframe)
        self.assertEqual(len(temporal.train), 1)
        self.assertEqual(len(temporal.validation), 1)
        self.assertEqual(len(temporal.test), 1)
        self.assertLess(
            temporal.train["arrival_date"].max(),
            temporal.validation["arrival_date"].min(),
        )
        self.assertLess(
            temporal.validation["arrival_date"].max(),
            temporal.test["arrival_date"].min(),
        )


@unittest.skipUnless(LOCAL_DATASET.is_file(), "Local CSV is not available")
class LocalDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataframe = load_dataset(LOCAL_DATASET)

    def test_known_raw_shape(self):
        self.assertEqual(self.dataframe.shape, (119_390, 32))

    def test_known_deduplication_and_temporal_counts(self):
        deduplicated, removed = remove_exact_duplicates(self.dataframe)
        temporal = split_by_arrival_date(deduplicated)
        self.assertEqual(removed, 31_994)
        self.assertEqual(len(temporal.train), 44_991)
        self.assertEqual(len(temporal.validation), 10_713)
        self.assertEqual(len(temporal.test), 31_692)


if __name__ == "__main__":
    unittest.main()
