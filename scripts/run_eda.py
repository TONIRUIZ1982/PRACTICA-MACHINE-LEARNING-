"""Generate a concise reproducible EDA summary without publishing data outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.hotel_cancellation.config import TARGET_COLUMN
from src.hotel_cancellation.data import load_dataset, remove_exact_duplicates


def build_eda_summary(dataframe: pd.DataFrame) -> dict:
    """Return only aggregate quality and target indicators for documentation."""
    missing = dataframe.isna().sum()
    return {
        "rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "exact_duplicates": int(dataframe.duplicated().sum()),
        "cancellation_rate": float(dataframe[TARGET_COLUMN].mean()),
        "missing_values": {
            column: int(count) for column, count in missing[missing > 0].items()
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/dataset_practica_final.csv"),
    )
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dataframe = load_dataset(arguments.data)
    raw_summary = build_eda_summary(dataframe)
    analyzed = dataframe
    if arguments.deduplicate:
        analyzed, removed = remove_exact_duplicates(dataframe)
        print(f"Duplicados exactos eliminados para el análisis: {removed:,}")

    summary = {"raw": raw_summary, "analyzed": build_eda_summary(analyzed)}
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "eda_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    target_counts = analyzed[TARGET_COLUMN].value_counts().sort_index()
    axis = target_counts.plot.bar(color=["#4C78A8", "#F58518"])
    axis.set_xticklabels(["No cancelada", "Cancelada"], rotation=0)
    axis.set_ylabel("Reservas")
    axis.set_title("Distribución de cancelaciones")
    plt.tight_layout()
    plt.savefig(arguments.output_dir / "target_distribution.png", dpi=150)
    plt.close()

    missing = analyzed.isna().sum()
    missing = missing[missing > 0].sort_values()
    if not missing.empty:
        axis = missing.plot.barh(color="#E45756")
        axis.set_xlabel("Valores ausentes")
        axis.set_title("Valores ausentes por variable")
        plt.tight_layout()
        plt.savefig(arguments.output_dir / "missing_values.png", dpi=150)
        plt.close()
    print(f"Resumen EDA local guardado en: {arguments.output_dir}")


if __name__ == "__main__":
    main()
