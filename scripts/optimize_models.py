"""Optimize candidate models, choose a winner and optionally evaluate 2017 once."""

from __future__ import annotations

import argparse
import json
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
from src.hotel_cancellation.modeling import evaluate_classifier, save_evaluation_artifacts
from src.hotel_cancellation.optimization import (
    best_f1_threshold,
    build_classical_searches,
    build_neural_candidates,
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
    parser.add_argument("--search-iterations", type=int, default=15)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--search-jobs", type=int, default=2)
    parser.add_argument("--neural-epochs", type=int, default=50)
    parser.add_argument("--neural-verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_optimization"),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evalúa una sola vez el ganador en el periodo final de 2017.",
    )
    return parser.parse_args()


def print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    print(f"\n{name}")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")
    print(f"  threshold: {metrics['threshold']:.4f}")


def main() -> None:
    arguments = parse_arguments()
    dataframe = load_dataset(arguments.data)
    if arguments.deduplicate:
        dataframe, removed = remove_exact_duplicates(dataframe)
        print(f"Duplicados exactos eliminados: {removed:,}")

    split = split_by_arrival_date(
        dataframe,
        train_end=arguments.train_end,
        validation_end=arguments.validation_end,
    )
    # TimeSeriesSplit only makes sense after ordering the training observations.
    ordered_train = split.train.sort_values("arrival_date", kind="stable")
    x_train, y_train = prepare_model_frame(ordered_train)
    x_validation, y_validation = prepare_model_frame(split.validation)
    print(f"Train: {len(x_train):,} | Validación: {len(x_validation):,}")

    fitted_models = {}
    results: dict[str, dict[str, float | int]] = {}
    details = {}
    searches = build_classical_searches(
        x_train,
        y_train,
        iterations=arguments.search_iterations,
        cv_splits=arguments.cv_splits,
        n_jobs=arguments.search_jobs,
    )
    for name, search in searches.items():
        print(f"\nOptimizando {name}...")
        started = perf_counter()
        search.fit(x_train, y_train)
        elapsed = perf_counter() - started
        model = search.best_estimator_
        probabilities = model.predict_proba(x_validation)[:, 1]
        threshold, _ = best_f1_threshold(y_validation, probabilities)
        metrics = evaluate_classifier(
            model, x_validation, y_validation, threshold=threshold
        )
        metrics["training_seconds"] = float(elapsed)
        metrics["cv_f1"] = float(search.best_score_)
        fitted_models[name] = model
        results[name] = metrics
        details[name] = {
            "best_parameters": search.best_params_,
            "cv_f1": float(search.best_score_),
            "threshold": threshold,
        }
        print_metrics(name, metrics)
        save_evaluation_artifacts(
            model,
            x_validation,
            y_validation,
            metrics,
            arguments.output_dir,
            prefix=f"optimized_{name}",
            threshold=threshold,
        )
        save_fitted_model(name, model, arguments.output_dir)

    neural_candidates = build_neural_candidates(
        x_train,
        epochs=arguments.neural_epochs,
        verbose=arguments.neural_verbose,
    )
    for name, model in neural_candidates.items():
        print(f"\nOptimizando {name}...")
        started = perf_counter()
        model.fit(x_train, y_train)
        elapsed = perf_counter() - started
        probabilities = model.predict_proba(x_validation)[:, 1]
        threshold, _ = best_f1_threshold(y_validation, probabilities)
        metrics = evaluate_classifier(
            model, x_validation, y_validation, threshold=threshold
        )
        metrics["training_seconds"] = float(elapsed)
        fitted_models[name] = model
        results[name] = metrics
        classifier = model.named_steps["classifier"]
        details[name] = {
            "hidden_units": classifier.hidden_units,
            "dropout_rate": classifier.dropout_rate,
            "epochs_completed": len(classifier.history_.history["loss"]),
            "threshold": threshold,
        }
        print_metrics(name, metrics)
        save_evaluation_artifacts(
            model,
            x_validation,
            y_validation,
            metrics,
            arguments.output_dir,
            prefix=f"optimized_{name}",
            threshold=threshold,
        )
        # Reuse the native Keras saver while isolating each architecture.
        save_fitted_model("neural_network", model, arguments.output_dir / name)

    table = save_comparison_table(results, arguments.output_dir)
    save_comparative_roc(
        fitted_models, x_validation, y_validation, arguments.output_dir
    )
    save_random_forest_importance(
        fitted_models["random_forest"], arguments.output_dir
    )
    with (arguments.output_dir / "optimization_details.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(details, file, indent=2, ensure_ascii=False, default=str)

    print("\nComparación optimizada:")
    print(table[["accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]].round(4))
    winner = select_best_model(results)
    winner_threshold = float(results[winner]["threshold"])
    print(f"\nGanador por F1 de validación: {winner}")

    if arguments.evaluate_test:
        x_test, y_test = prepare_model_frame(split.test)
        test_metrics = evaluate_classifier(
            fitted_models[winner],
            x_test,
            y_test,
            threshold=winner_threshold,
        )
        print_metrics(f"TEST FINAL 2017 - {winner}", test_metrics)
        save_evaluation_artifacts(
            fitted_models[winner],
            x_test,
            y_test,
            test_metrics,
            arguments.output_dir,
            prefix=f"final_test_{winner}",
            threshold=winner_threshold,
        )
        with (arguments.output_dir / "final_test_result.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {"winner": winner, "metrics": test_metrics},
                file,
                indent=2,
                ensure_ascii=False,
            )
        print("El test final ya ha sido consumido; no debe usarse para nuevos ajustes.")
    else:
        print("Test 2017 reservado: use --evaluate-test solo al cerrar el modelo.")


if __name__ == "__main__":
    main()
