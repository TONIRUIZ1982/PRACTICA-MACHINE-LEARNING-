"""Factories for the five classification models required by the assignment."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .modeling import build_logistic_pipeline, build_preprocessor


MODEL_ORDER = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "neural_network",
)


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible Keras multilayer binary classifier."""

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.20,
        epochs: int = 30,
        batch_size: int = 256,
        patience: int = 4,
        validation_split: float = 0.15,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, features, target):
        """Fit with a stratified training-only split for early stopping."""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        features = np.asarray(features, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        # Keras otherwise reserves the last rows before shuffling. The source
        # data is ordered by hotel and time, so a stratified training-only split
        # is safer and never touches the external validation period.
        fit_features, stop_features, fit_target, stop_target = train_test_split(
            features,
            target,
            test_size=self.validation_split,
            stratify=target,
            random_state=self.random_state,
        )

        layers = [tf.keras.layers.Input(shape=(features.shape[1],))]
        for units in self.hidden_units:
            layers.extend(
                [
                    tf.keras.layers.Dense(units, activation="relu"),
                    tf.keras.layers.Dropout(self.dropout_rate),
                ]
            )
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model_ = tf.keras.Sequential(layers)
        self.model_.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        self.history_ = self.model_.fit(
            fit_features,
            fit_target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(stop_features, stop_target),
            callbacks=[callback],
            shuffle=True,
            verbose=self.verbose,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features) -> np.ndarray:
        """Return two columns to match sklearn's probability convention."""
        probabilities = self.model_.predict(
            np.asarray(features, dtype=np.float32),
            batch_size=self.batch_size,
            verbose=0,
        ).reshape(-1)
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, features) -> np.ndarray:
        """Predict labels using the neutral 0.5 threshold."""
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def build_required_models(
    predictors: pd.DataFrame,
    *,
    neural_epochs: int = 30,
    neural_verbose: int = 0,
) -> dict[str, Pipeline]:
    """Build the five mandatory models with fixed reproducible baselines."""
    models: dict[str, Pipeline] = {
        "logistic_regression": build_logistic_pipeline(predictors),
        "decision_tree": Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(predictors)),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(predictors)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(predictors)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "neural_network": Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(predictors, dense_output=True)),
                (
                    "classifier",
                    KerasBinaryClassifier(
                        epochs=neural_epochs,
                        verbose=neural_verbose,
                    ),
                ),
            ]
        ),
    }
    return {name: models[name] for name in MODEL_ORDER}
