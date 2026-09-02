"""Feature engineering shared by training scripts and future notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET_COLUMN
from .data import DatasetValidationError, get_candidate_features


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
