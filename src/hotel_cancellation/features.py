"""Feature engineering shared by training, evaluation and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN
from .data import DatasetValidationError


# Final baseline exclusions. ``agent`` adds hundreds of unstable categories
# without improving temporal validation. The day of month also showed no useful
# incremental value. Week is represented below using cyclical coordinates.
MODEL_ONLY_EXCLUSIONS = (
    "arrival_date",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "agent",
)

# Exact predictor order used by the trained pipelines. Keeping this explicit
# prevents training/inference drift when a future data source has extra fields.
MODEL_PREDICTOR_COLUMNS = (
    "hotel",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_children",
    "arrival_week_sin",
    "arrival_week_cos",
)

# Fields a new reservation must supply. Target, leakage columns and variables
# unavailable at booking time are deliberately absent from this contract.
INFERENCE_INPUT_COLUMNS = (
    "hotel",
    "lead_time",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the selected interpretable booking features."""
    prepared = dataframe.copy()

    children = prepared["children"].fillna(0)
    prepared["has_children"] = ((children + prepared["babies"]) > 0).astype(int)

    # Week captures seasonality at a finer resolution than month. Sine/cosine
    # preserve the fact that the beginning and end of the year are neighbours.
    prepared["arrival_week_sin"] = np.sin(
        2 * np.pi * prepared["arrival_date_week_number"] / 53
    )
    prepared["arrival_week_cos"] = np.cos(
        2 * np.pi * prepared["arrival_date_week_number"] / 53
    )
    return prepared


def prepare_model_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the fixed predictors and target used for model development."""
    if TARGET_COLUMN not in dataframe:
        raise DatasetValidationError(f"Missing target column: {TARGET_COLUMN}.")
    prepared = add_engineered_features(dataframe)
    missing = sorted(set(MODEL_PREDICTOR_COLUMNS) - set(prepared.columns))
    if missing:
        raise DatasetValidationError(f"Missing model predictor columns: {missing}.")
    return (
        prepared.loc[:, MODEL_PREDICTOR_COLUMNS].copy(),
        prepared[TARGET_COLUMN].copy(),
    )


def prepare_inference_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build exactly the same predictors for new reservations without target."""
    missing = sorted(set(INFERENCE_INPUT_COLUMNS) - set(dataframe.columns))
    if missing:
        raise DatasetValidationError(f"Missing inference input columns: {missing}.")
    prepared = add_engineered_features(dataframe)
    return prepared.loc[:, MODEL_PREDICTOR_COLUMNS].copy()
