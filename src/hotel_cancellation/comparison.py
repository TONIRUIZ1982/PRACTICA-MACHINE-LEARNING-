"""Evaluation, comparison and persistence of fitted classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

# Comparative charts are files, so no desktop GUI event loop is required.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay

from .modeling import PRIMARY_METRIC


def select_best_model(
    results: dict[str, dict[str, float | int]],
    *,
    metric: str = PRIMARY_METRIC,
) -> str:
    """Select the greatest validation score using the agreed main metric."""
    if not results:
        raise ValueError("No model results were provided.")
    missing = [name for name, values in results.items() if metric not in values]
    if missing:
        raise ValueError(f"Metric '{metric}' is missing for models: {missing}.")
    return max(results, key=lambda name: float(results[name][metric]))


def save_comparison_table(
    results: dict[str, dict[str, float | int]],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Save common validation metrics in machine- and human-readable formats."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame.from_dict(results, orient="index")
    table.index.name = "model"
    table = table.sort_values(PRIMARY_METRIC, ascending=False)
    table.to_csv(output_path / "model_comparison.csv")
    with (output_path / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    return table


def save_comparative_roc(
    models: dict,
    predictors: pd.DataFrame,
    target: pd.Series,
    output_directory: str | Path,
) -> Path:
    """Draw every validation ROC curve on the same axes."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        probabilities = model.predict_proba(predictors)[:, 1]
        RocCurveDisplay.from_predictions(target, probabilities, name=name, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Azar")
    axis.set_title("Curvas ROC comparativas - validación")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure_path = output_path / "comparative_roc.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def save_random_forest_importance(
    random_forest_pipeline,
    output_directory: str | Path,
    *,
    top_n: int = 25,
) -> Path:
    """Save the Random Forest feature-importance table and readable chart."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    preprocessor = random_forest_pipeline.named_steps["preprocessing"]
    classifier = random_forest_pipeline.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_path / "random_forest_feature_importance.csv", index=False)

    plotted = importance.head(top_n).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(plotted["feature"], plotted["importance"])
    axis.set_title(f"Random Forest - {top_n} variables más importantes")
    axis.set_xlabel("Importancia")
    figure.tight_layout()
    figure_path = output_path / "random_forest_feature_importance.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def save_fitted_model(name: str, model, output_directory: str | Path) -> None:
    """Persist sklearn pipelines and Keras components in their native formats."""
    output_path = Path(output_directory) / "models"
    output_path.mkdir(parents=True, exist_ok=True)
    if name == "neural_network":
        joblib.dump(
            model.named_steps["preprocessing"],
            output_path / "neural_network_preprocessing.joblib",
        )
        model.named_steps["classifier"].model_.save(
            output_path / "neural_network.keras"
        )
    else:
        joblib.dump(model, output_path / f"{name}.joblib")
