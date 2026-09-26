"""Inference helpers for the final XGBoost cancellation model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .features import prepare_inference_frame


FINAL_MODEL_NAME = "xgboost"
FINAL_THRESHOLD = 0.3618723452091217


def load_final_model(path: str | Path):
    """Load the local XGBoost pipeline selected after temporal validation."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Final model not found at '{model_path}'. Train it locally first."
        )
    return joblib.load(model_path)


def predict_cancellations(
    model,
    reservations: pd.DataFrame,
    *,
    threshold: float = FINAL_THRESHOLD,
) -> pd.DataFrame:
    """Return cancellation probabilities and decisions for new reservations."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    predictors = prepare_inference_frame(reservations)
    probabilities = model.predict_proba(predictors)[:, 1]
    return pd.DataFrame(
        {
            "cancellation_probability": probabilities,
            "predicted_is_canceled": (probabilities >= threshold).astype(int),
            "decision_threshold": threshold,
        },
        index=reservations.index,
    )
