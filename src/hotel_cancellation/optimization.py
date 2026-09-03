"""Temporal hyperparameter and threshold optimization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .model_suite import KerasBinaryClassifier, build_required_models
from .modeling import build_preprocessor


def best_f1_threshold(target, probabilities) -> tuple[float, float]:
    """Find the validation threshold that maximizes F1.

    Ties are resolved in favour of the value closest to 0.5 so that the chosen
    decision rule remains as conservative and stable as possible.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    maximum = float(scores.max())
    candidate_indices = np.flatnonzero(np.isclose(scores, maximum))
    best_index = candidate_indices[
        np.argmin(np.abs(thresholds[candidate_indices] - 0.5))
    ]
    return float(thresholds[best_index]), maximum


def build_classical_searches(
    predictors: pd.DataFrame,
    target: pd.Series,
    *,
    iterations: int = 15,
    cv_splits: int = 4,
    n_jobs: int = 2,
) -> dict[str, RandomizedSearchCV]:
    """Build reproducible randomized searches with chronological CV folds."""
    if iterations < 1:
        raise ValueError("iterations must be positive.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    models = build_required_models(predictors, neural_epochs=1)
    models.pop("neural_network")
    # Avoid nested parallelism: the search owns the workers.
    models["random_forest"].set_params(classifier__n_jobs=1)
    models["xgboost"].set_params(classifier__n_jobs=1)
    positive_weight = float((target == 0).sum() / (target == 1).sum())

    spaces = {
        "logistic_regression": {
            "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "decision_tree": {
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__max_depth": [4, 6, 8, 10, 12, 16, None],
            "classifier__min_samples_split": [2, 10, 25, 50, 100],
            "classifier__min_samples_leaf": [1, 5, 10, 20, 50],
            "classifier__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [8, 12, 16, 24, None],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", 0.5],
            "classifier__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "xgboost": {
            "classifier__n_estimators": [200, 350, 500, 700],
            "classifier__max_depth": [3, 4, 5, 6, 8],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "classifier__min_child_weight": [1, 3, 5, 10],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "classifier__reg_alpha": [0.0, 0.01, 0.1, 0.5],
            "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "classifier__scale_pos_weight": [
                1.0,
                positive_weight * 0.75,
                positive_weight,
                positive_weight * 1.25,
            ],
        },
    }
    cross_validation = TimeSeriesSplit(n_splits=cv_splits)
    return {
        name: RandomizedSearchCV(
            estimator=model,
            param_distributions=spaces[name],
            n_iter=iterations,
            scoring="f1",
            cv=cross_validation,
            refit=True,
            random_state=42,
            n_jobs=n_jobs,
            verbose=1,
            error_score="raise",
        )
        for name, model in models.items()
    }


def build_neural_candidates(
    predictors: pd.DataFrame,
    *,
    epochs: int = 50,
    verbose: int = 0,
) -> dict[str, Pipeline]:
    """Build a small, bounded architecture comparison for Keras."""
    configurations = {
        "neural_network_small": ((64, 32), 0.20),
        "neural_network_wide": ((128, 64), 0.25),
        "neural_network_deep": ((128, 64, 32), 0.30),
    }
    return {
        name: Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(predictors, dense_output=True)),
                (
                    "classifier",
                    KerasBinaryClassifier(
                        hidden_units=hidden_units,
                        dropout_rate=dropout_rate,
                        epochs=epochs,
                        patience=5,
                        verbose=verbose,
                    ),
                ),
            ]
        )
        for name, (hidden_units, dropout_rate) in configurations.items()
    }
