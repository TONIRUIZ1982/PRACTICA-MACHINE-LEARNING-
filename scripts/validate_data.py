"""Validate the local dataset and report the proposed temporal split."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.hotel_cancellation import (
    TARGET_COLUMN,
    get_candidate_features,
    load_dataset,
    remove_exact_duplicates,
    split_by_arrival_date,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/dataset_practica_final.csv"),
        help="Path to the local CSV.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Report the proposed split after removing exact duplicates.",
    )
    parser.add_argument("--train-end", default="2016-09-30")
    parser.add_argument("--validation-end", default="2016-12-31")
    return parser.parse_args()


def cancellation_rate(dataframe) -> float:
    return float(dataframe[TARGET_COLUMN].mean())


def main() -> None:
    arguments = parse_arguments()
    dataframe = load_dataset(arguments.data)
    print(f"Loaded rows: {len(dataframe):,}")

    if arguments.deduplicate:
        dataframe, removed = remove_exact_duplicates(dataframe)
        print(f"Removed exact duplicates: {removed:,}")

    temporal = split_by_arrival_date(
        dataframe,
        train_end=arguments.train_end,
        validation_end=arguments.validation_end,
    )
    for name, subset in (
        ("train", temporal.train),
        ("validation", temporal.validation),
        ("test", temporal.test),
    ):
        print(
            f"{name}: {len(subset):,} rows; "
            f"cancellation rate={cancellation_rate(subset):.2%}"
        )

    candidates = get_candidate_features(dataframe)
    print(f"Initial candidate features: {len(candidates)}")


if __name__ == "__main__":
    main()
