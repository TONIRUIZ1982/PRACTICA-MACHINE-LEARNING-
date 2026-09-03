"""Single entry point for validation, EDA, training, optimization and inference."""

from __future__ import annotations

import argparse
import subprocess
import sys


COMMAND_MODULES = {
    "validate": "scripts.validate_data",
    "eda": "scripts.run_eda",
    "baseline": "scripts.train_logistic_baseline",
    "compare": "scripts.train_all_models",
    "optimize": "scripts.optimize_models",
    "predict": "scripts.predict_cancellations",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMAND_MODULES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    forwarded = arguments.arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    completed = subprocess.run(
        [sys.executable, "-m", COMMAND_MODULES[arguments.command], *forwarded],
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
