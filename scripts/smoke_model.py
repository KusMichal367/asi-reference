"""Smoke modelu (Sprint 3): pobranie → wczytanie → predykcja 5 wierszy → kontrola schematu.

Uruchamiany w osobnym procesie, bez treningu:
    python scripts/smoke_model.py --artifact ENTITY/PROJEKT/churn-model:production
    python scripts/smoke_model.py --package-dir data/06_models/package   # bez W&B
"""

import argparse
import sys

import pandas as pd

from asi_project.config import load_project_config
from asi_project.model_package import download_package_artifact, load_package, predict

N_ROWS = 5


def check_schema(result: pd.DataFrame, n_rows: int) -> list[str]:
    problems = []
    if list(result.columns) != ["prediction", "probability"]:
        problems.append(f"zle kolumny: {list(result.columns)}")
    if len(result) != n_rows:
        problems.append(f"zla liczba wierszy: {len(result)} zamiast {n_rows}")
    if not set(result.get("prediction", [])) <= {0, 1}:
        problems.append("prediction poza {0, 1}")
    if "probability" in result and not result["probability"].between(0, 1).all():
        problems.append("probability poza [0, 1]")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", help="np. entity/projekt/churn-model:production")
    source.add_argument("--package-dir", help="lokalny katalog pakietu")
    args = parser.parse_args(argv)

    if args.artifact:
        package_dir = download_package_artifact(args.artifact)
        print(f"Pobrano: {args.artifact} -> {package_dir}")
    else:
        package_dir = args.package_dir

    model, metadata = load_package(package_dir)
    print(f"Wczytano: {metadata['model_name']} (git {metadata['git_sha'][:7]})")

    rows = pd.read_csv(load_project_config().ci_sample_path).head(N_ROWS)
    result = predict(model, metadata, rows)
    print(result.to_string())

    problems = check_schema(result, N_ROWS)
    if problems:
        print("SMOKE FAILED: " + "; ".join(problems))
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
