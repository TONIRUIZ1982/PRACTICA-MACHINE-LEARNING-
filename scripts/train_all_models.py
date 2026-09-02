"""Train, compare and select the five classifiers required by the assignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from src.hotel_cancellation.comparison import (
    save_comparison_table,
    save_comparative_roc,
    save_fitted_model,
    save_random_forest_importance,
    select_best_model,
)
from src.hotel_cancellation.data import (
    load_dataset,
    remove_exact_duplicates,
    split_by_arrival_date,
)
from src.hotel_cancellation.features import prepare_model_frame
from src.hotel_cancellation.model_suite import MODEL_ORDER, build_required_models
from src.hotel_cancellation.modeling import (
    PRIMARY_METRIC,
    evaluate_classifier,
    save_evaluation_artifacts,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/dataset_practica_final.csv"),
    )
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--train-end", default="2016-09-30")
    parser.add_argument("--validation-end", default="2016-12-31")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neural-epochs", type=int, default=30)
    parser.add_argument("--neural-verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Por defecto entrena los cinco modelos obligatorios.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_comparison"),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evalúa solo el ganador en 2017. Usar una única vez al final.",
    )
    return parser.parse_args()


def print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    print(f"\n{name}")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")


def main() -> None:
    arguments = parse_arguments()
    if not 0 < arguments.threshold < 1:
        raise ValueError("--threshold debe estar entre 0 y 1.")
    if arguments.neural_epochs < 1:
        raise ValueError("--neural-epochs debe ser positivo.")

    dataframe = load_dataset(arguments.data)
    if arguments.deduplicate:
        dataframe, removed = remove_exact_duplicates(dataframe)
        print(f"Duplicados exactos eliminados: {removed:,}")
    split = split_by_arrival_date(
        dataframe,
        train_end=arguments.train_end,
        validation_end=arguments.validation_end,
    )
    x_train, y_train = prepare_model_frame(split.train)
    x_validation, y_validation = prepare_model_frame(split.validation)
    print(f"Train: {len(x_train):,} | Validación: {len(x_validation):,}")
    print(f"Variables: {x_train.shape[1]} | Métrica principal: {PRIMARY_METRIC}")

    available_models = build_required_models(
        x_train,
        neural_epochs=arguments.neural_epochs,
        neural_verbose=arguments.neural_verbose,
    )
    models = {name: available_models[name] for name in arguments.models}
    results: dict[str, dict[str, float | int]] = {}
    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        started = perf_counter()
        model.fit(x_train, y_train)
        elapsed = perf_counter() - started
        metrics = evaluate_classifier(
            model,
            x_validation,
            y_validation,
            threshold=arguments.threshold,
        )
        metrics["training_seconds"] = float(elapsed)
        results[name] = metrics
        print_metrics(name, metrics)
        save_evaluation_artifacts(
            model,
            x_validation,
            y_validation,
            metrics,
            arguments.output_dir,
            prefix=f"validation_{name}",
            threshold=arguments.threshold,
        )
        save_fitted_model(name, model, arguments.output_dir)

    table = save_comparison_table(results, arguments.output_dir)
    save_comparative_roc(models, x_validation, y_validation, arguments.output_dir)
    if "random_forest" in models:
        save_random_forest_importance(models["random_forest"], arguments.output_dir)
    print("\nComparación de validación:")
    print(table[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4))

    best_name = select_best_model(results)
    print(f"\nMejor modelo por {PRIMARY_METRIC}: {best_name}")
    if arguments.evaluate_test:
        # Only the validation winner is exposed to the untouched final period.
        x_test, y_test = prepare_model_frame(split.test)
        test_metrics = evaluate_classifier(
            models[best_name],
            x_test,
            y_test,
            threshold=arguments.threshold,
        )
        print_metrics(f"Test final - {best_name}", test_metrics)
        save_evaluation_artifacts(
            models[best_name],
            x_test,
            y_test,
            test_metrics,
            arguments.output_dir,
            prefix=f"test_{best_name}",
            threshold=arguments.threshold,
        )
    else:
        print("Test 2017 reservado: no se ha evaluado ningún modelo.")


if __name__ == "__main__":
    main()
