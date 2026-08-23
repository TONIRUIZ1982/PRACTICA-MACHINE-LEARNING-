"""Loading, validation and temporal splitting of hotel booking data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    EXPECTED_COLUMNS,
    HIGH_MISSING_FEATURES,
    LEAKAGE_FEATURES,
    MONTH_TO_NUMBER,
    TARGET_COLUMN,
    TIME_DEPENDENT_FEATURES,
)


class DatasetValidationError(ValueError):
    """Raised when the source dataset does not satisfy the agreed schema."""


@dataclass(frozen=True)
class TemporalSplit:
    """Non-overlapping train, validation and test periods."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the original CSV and validate its schema and binary target."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Place it locally under data/raw/."
        )

    dataframe = pd.read_csv(csv_path, na_values=["NULL"])
    validate_dataset(dataframe)
    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Validate raw columns and target values without changing the data."""
    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise DatasetValidationError(f"Duplicated column names: {duplicates}")

    actual_columns = set(dataframe.columns)
    expected_columns = set(EXPECTED_COLUMNS)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if missing or unexpected:
        raise DatasetValidationError(
            f"Invalid dataset schema. Missing={missing}; unexpected={unexpected}."
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise DatasetValidationError("The target contains missing values.")

    target_values = set(dataframe[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise DatasetValidationError(
            f"The target must be binary (0/1); found {sorted(target_values)}."
        )


def build_arrival_date(dataframe: pd.DataFrame) -> pd.Series:
    """Build the planned arrival date without using outcome-related dates."""
    required = {
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise DatasetValidationError(
            f"Cannot build arrival date; missing columns: {missing}."
        )

    month_numbers = dataframe["arrival_date_month"].map(MONTH_TO_NUMBER)
    unknown_months = sorted(
        dataframe.loc[month_numbers.isna(), "arrival_date_month"]
        .dropna()
        .astype(str)
        .unique()
    )
    if month_numbers.isna().any():
        raise DatasetValidationError(
            f"Unknown or missing arrival months: {unknown_months}."
        )

    try:
        arrival_date = pd.to_datetime(
            {
                "year": dataframe["arrival_date_year"],
                "month": month_numbers,
                "day": dataframe["arrival_date_day_of_month"],
            },
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise DatasetValidationError("Invalid planned arrival date.") from error

    arrival_date.name = "arrival_date"
    return arrival_date


def remove_exact_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return a deduplicated copy and the number of removed exact rows."""
    deduplicated = dataframe.drop_duplicates().copy()
    return deduplicated, len(dataframe) - len(deduplicated)


def get_candidate_features(
    dataframe: pd.DataFrame,
    *,
    include_time_dependent: bool = False,
    include_company: bool = False,
    extra_exclusions: Iterable[str] = (),
) -> list[str]:
    """Return predictors while always excluding the target and leakage."""
    excluded = {TARGET_COLUMN, *LEAKAGE_FEATURES, *extra_exclusions}
    if not include_time_dependent:
        excluded.update(TIME_DEPENDENT_FEATURES)
    if not include_company:
        excluded.update(HIGH_MISSING_FEATURES)
    return [column for column in dataframe.columns if column not in excluded]


def split_by_arrival_date(
    dataframe: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp = "2016-09-30",
    validation_end: str | pd.Timestamp = "2016-12-31",
) -> TemporalSplit:
    """Split chronologically using the planned arrival date.

    Dates up to ``train_end`` form the training set. Later dates up to
    ``validation_end`` form validation, and the remaining future dates form
    the final test set.
    """
    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be earlier than validation_end.")

    arrival_date = build_arrival_date(dataframe)
    prepared = dataframe.copy()
    prepared["arrival_date"] = arrival_date

    train = prepared.loc[arrival_date <= train_boundary].copy()
    validation = prepared.loc[
        (arrival_date > train_boundary) & (arrival_date <= validation_boundary)
    ].copy()
    test = prepared.loc[arrival_date > validation_boundary].copy()

    empty_sets = [
        name
        for name, subset in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        if subset.empty
    ]
    if empty_sets:
        raise DatasetValidationError(
            f"Temporal boundaries produced empty sets: {empty_sets}."
        )

    return TemporalSplit(train=train, validation=validation, test=test)
