"""Run non-invasive checks before the final project delivery.

The command never retrains models and never evaluates the held-out 2017 test set.
It only validates that the repository is safe to share and, optionally, that the
local CSV and selected model are available.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.hotel_cancellation import load_dataset
from src.hotel_cancellation.inference import load_final_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "docs/VARIABLES.md",
    "docs/DECISIONES_DATOS.md",
    "docs/RESULTADO_FINAL.md",
    "docs/INFORME_FINAL_BORRADOR.md",
    "docs/GUIA_DEFENSA.md",
    "docs/ENTREGA_PENDIENTE.md",
)
PRIVATE_PREFIXES = ("data/raw/", "artifacts/", "models/")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/dataset_practica_final.csv"),
        help="Local CSV to validate when it exists.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model_optimization/models/xgboost.joblib"),
        help="Local selected model to verify when it exists.",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail if the local CSV is unavailable.",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if the local selected model is unavailable.",
    )
    return parser.parse_args()


def tracked_private_paths(root: Path) -> list[str]:
    """Return tracked files that must stay local rather than in Git."""
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        path
        for path in completed.stdout.splitlines()
        if path.startswith(PRIVATE_PREFIXES)
    ]


def existing_required_files(root: Path) -> list[str]:
    return [path for path in REQUIRED_FILES if not (root / path).is_file()]


def main() -> None:
    arguments = parse_arguments()
    errors: list[str] = []

    missing_files = existing_required_files(PROJECT_ROOT)
    if missing_files:
        errors.append("Faltan documentos: " + ", ".join(missing_files))
    else:
        print(f"[OK] Documentacion de cierre: {len(REQUIRED_FILES)} archivos")

    private_paths = tracked_private_paths(PROJECT_ROOT)
    if private_paths:
        errors.append(
            "Hay datos, artefactos o modelos versionados: "
            + ", ".join(private_paths)
        )
    else:
        print("[OK] No hay datos, modelos ni artefactos locales versionados")

    data_path = arguments.data
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if data_path.is_file():
        dataframe = load_dataset(data_path)
        print(f"[OK] CSV local valido: {len(dataframe):,} filas")
    elif arguments.require_data:
        errors.append(f"No se encontro el CSV local: {data_path}")
    else:
        print("[INFO] CSV local no disponible: comprobacion omitida")

    model_path = arguments.model
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if model_path.is_file():
        load_final_model(model_path)
        print("[OK] Modelo final local disponible")
    elif arguments.require_model:
        errors.append(f"No se encontro el modelo local: {model_path}")
    else:
        print("[INFO] Modelo local no disponible: comprobacion omitida")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)

    print("[OK] Comprobacion de entrega completada")


if __name__ == "__main__":
    main()
