"""Feature engineering shared by training scripts and future notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MONTH_TO_NUMBER, TARGET_COLUMN
from .data import DatasetValidationError, get_candidate_features


# These columns locate a booking in time, but should not be learnt as arbitrary
# categories. Month and week are represented below with cyclical coordinates.
MODEL_ONLY_EXCLUSIONS = (
    "arrival_date",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with compact, interpretable booking features."""
    prepared = dataframe.copy()

    prepared["total_nights"] = (
        prepared["stays_in_weekend_nights"] + prepared["stays_in_week_nights"]
    )
    children = prepared["children"].fillna(0)
    prepared["total_guests"] = prepared["adults"] + children + prepared["babies"]
    prepared["has_children"] = ((children + prepared["babies"]) > 0).astype(int)
    prepared["has_previous_bookings"] = (
        (
            prepared["previous_cancellations"]
            + prepared["previous_bookings_not_canceled"]
        )
        > 0
    ).astype(int)

    month_number = prepared["arrival_date_month"].map(MONTH_TO_NUMBER)
    if month_number.isna().any():
        unknown = sorted(
            prepared.loc[month_number.isna(), "arrival_date_month"]
            .dropna()
            .astype(str)
            .unique()
        )
        raise DatasetValidationError(f"Unknown or missing arrival months: {unknown}.")

    # Sine/cosine preserve the fact that December and January are neighbours.
    prepared["arrival_month_sin"] = np.sin(2 * np.pi * month_number / 12)
    prepared["arrival_month_cos"] = np.cos(2 * np.pi * month_number / 12)
    prepared["arrival_week_sin"] = np.sin(
        2 * np.pi * prepared["arrival_date_week_number"] / 53
    )
    prepared["arrival_week_cos"] = np.cos(
        2 * np.pi * prepared["arrival_date_week_number"] / 53
    )
    return prepared


def prepare_model_frame(
    dataframe: pd.DataFrame,
    *,
    include_time_dependent: bool = False,
    include_company: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build predictors and target using the project's agreed exclusions."""
    if TARGET_COLUMN not in dataframe:
        raise DatasetValidationError(f"Missing target column: {TARGET_COLUMN}.")

    prepared = add_engineered_features(dataframe)
    feature_names = get_candidate_features(
        prepared,
        include_time_dependent=include_time_dependent,
        include_company=include_company,
        extra_exclusions=MODEL_ONLY_EXCLUSIONS,
    )
    return prepared.loc[:, feature_names].copy(), prepared[TARGET_COLUMN].copy()
