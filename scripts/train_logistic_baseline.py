"""Train and validate the first reproducible cancellation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.hotel_cancellation.data import (
    load_dataset,
    remove_exact_duplicates,
    split_by_arrival_date,
)
from src.hotel_cancellation.features import prepare_model_frame
from src.hotel_cancellation.modeling import (
    PRIMARY_METRIC,
    build_logistic_pipeline,
    evaluate_classifier,
    save_evaluation_artifacts,
    save_model,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/dataset_practica_final.csv"),
        help="Ruta al CSV local (nunca se publica en Git).",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Elimina duplicados exactos antes de separar los periodos.",
    )
    parser.add_argument("--train-end", default="2016-09-30")
    parser.add_argument("--validation-end", default="2016-12-31")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/logistic_baseline"),
        help="Carpeta local ignorada por Git para métricas, gráficos y modelo.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evalúa 2017. Usar solo tras cerrar todas las decisiones del modelo.",
    )
    return parser.parse_args()


def print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    print(f"\n{name} ({metrics['samples']:,} filas)")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")


def evaluate_and_save(model, predictors, target, arguments, prefix: str):
    metrics = evaluate_classifier(
        model,
        predictors,
        target,
        threshold=arguments.threshold,
    )
    save_evaluation_artifacts(
        model,
        predictors,
        target,
        metrics,
        arguments.output_dir,
        prefix=prefix,
        threshold=arguments.threshold,
    )
    print_metrics(prefix.capitalize(), metrics)
    return metrics


def main() -> None:
    arguments = parse_arguments()
    if not 0 < arguments.threshold < 1:
        raise ValueError("--threshold debe estar entre 0 y 1.")

    dataframe = load_dataset(arguments.data)
    print(f"Filas cargadas: {len(dataframe):,}")
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

    # El pipeline aprende imputación, escalado y categorías solo con train.
    model = build_logistic_pipeline(x_train)
    model.fit(x_train, y_train)
    print(f"Variables de entrada: {x_train.shape[1]}")
    print(f"Métrica principal acordada para comparar modelos: {PRIMARY_METRIC}")
    evaluate_and_save(
        model, x_validation, y_validation, arguments, prefix="validation"
    )
    model_path = save_model(model, arguments.output_dir)
    print(f"\nModelo local guardado en: {model_path}")

    if arguments.evaluate_test:
        # Esta ruta queda deliberadamente separada para proteger el test final.
        x_test, y_test = prepare_model_frame(split.test)
        evaluate_and_save(model, x_test, y_test, arguments, prefix="test")
    else:
        print("Test 2017 reservado: no se ha calculado ninguna métrica final.")


if __name__ == "__main__":
    main()
