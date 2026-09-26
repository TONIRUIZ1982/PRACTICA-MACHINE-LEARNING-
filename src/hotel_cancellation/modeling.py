"""Reusable preprocessing, baseline model and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

# Scripts run headlessly; a GUI backend can crash when estimators use workers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRIMARY_METRIC = "f1"


def build_preprocessor(
    predictors: pd.DataFrame,
    *,
    dense_output: bool = False,
) -> ColumnTransformer:
    """Create preprocessing from training dtypes only.

    Dense output is reserved for Keras. Scikit-learn and XGBoost keep the
    memory-efficient sparse one-hot representation.
    """
    numeric_columns = predictors.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in predictors.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
        sparse_threshold=0.0 if dense_output else 0.3,
    )


def build_logistic_pipeline(predictors: pd.DataFrame) -> Pipeline:
    """Create an end-to-end logistic baseline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor(predictors)),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=42),
            ),
        ]
    )


def evaluate_classifier(
    model,
    predictors: pd.DataFrame,
    target: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate the same classification metrics for every model."""
    probabilities = model.predict_proba(predictors)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "samples": int(len(target)),
        "positive_rate": float(target.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def save_evaluation_artifacts(
    model,
    predictors: pd.DataFrame,
    target: pd.Series,
    metrics: dict[str, float | int],
    output_directory: str | Path,
    *,
    prefix: str,
    threshold: float = 0.5,
) -> None:
    """Save local-only metrics and plots; the directory is ignored by Git."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(predictors)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    with (output_path / f"{prefix}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    ConfusionMatrixDisplay.from_predictions(
        target,
        predictions,
        display_labels=["No cancelada", "Cancelada"],
        colorbar=False,
    )
    plt.title(f"Matriz de confusión - {prefix}")
    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(target, probabilities)
    plt.title(f"Curva ROC - {prefix}")
    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}_roc_curve.png", dpi=150)
    plt.close()


def save_model(model: Pipeline, output_directory: str | Path) -> Path:
    """Persist the fitted logistic pipeline for local reproducibility."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "logistic_baseline.joblib"
    joblib.dump(model, model_path)
    return model_path
