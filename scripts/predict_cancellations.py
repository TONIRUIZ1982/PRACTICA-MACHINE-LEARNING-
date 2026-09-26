"""Predict hotel booking cancellations with the selected local XGBoost model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.hotel_cancellation.inference import (
    FINAL_THRESHOLD,
    load_final_model,
    predict_cancellations,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV de reservas nuevas.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model_optimization/models/xgboost.joblib"),
        help="Ruta local al pipeline XGBoost entrenado.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/predictions/cancellation_predictions.csv"),
        help="CSV local con probabilidades y decisiones.",
    )
    parser.add_argument("--threshold", type=float, default=FINAL_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    reservations = pd.read_csv(arguments.input, na_values=["NULL"])
    model = load_final_model(arguments.model)
    predictions = predict_cancellations(
        model,
        reservations,
        threshold=arguments.threshold,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(arguments.output, index=False)
    print(f"Predicciones creadas: {len(predictions):,}")
    print(f"Probabilidad media: {predictions['cancellation_probability'].mean():.2%}")
    print(f"Salida local: {arguments.output}")


if __name__ == "__main__":
    main()
