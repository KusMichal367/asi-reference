"""Buduje pakiet modelu z wyników pipeline'u i (opcjonalnie) wysyła go do W&B."""

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

from asi_project.config import get_contract, load_project_config
from asi_project.data import raw_data_path
from asi_project.metrics import load_metrics
from asi_project.model_package import (
    build_metadata,
    file_fingerprint,
    log_package_artifact,
    save_package,
)

MODEL_FILES = {
    "logistic_regression": "data/06_models/trained_model.pkl",
    "dummy": "data/06_models/dummy_model.pkl",
}


def load_model(path: Path) -> tuple[Any, str]:
    """Katalog = TabularPredictor (AutoGluon), plik = model sklearn (pickle z Kedro)."""

    if path.is_dir():
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(path)), "autogluon"
    with open(path, "rb") as handle:
        return pickle.load(handle), "sklearn"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="plik .pkl albo katalog AutoGluona (domyślnie wg --model-name)"
    )
    parser.add_argument("--metrics", default="data/08_reporting/metrics.json")
    parser.add_argument("--model-name", default="logistic_regression")
    parser.add_argument("--out", default="data/06_models/package")
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT"),
        help="domyślnie WANDB_PROJECT; bez projektu nie trafi nic do W&B",
    )
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--alias", action="append", default=None)
    args = parser.parse_args(argv)

    config = load_project_config()
    contract = get_contract(config)

    model_path = args.model or MODEL_FILES.get(args.model_name)
    if model_path is None:
        parser.error(f"podaj --model dla modelu {args.model_name}")
    model, model_format = load_model(Path(model_path))

    record = load_metrics(args.metrics)
    if args.model_name not in record["models"]:
        parser.error(f"brak metryk modelu {args.model_name} w {args.metrics}")
    if list(record["features"]) != list(config.model_features):
        parser.error("cechy w metrics.json różnią się od project.yaml - uruchom trening ponownie")

    if record["data_mode"] == "full":
        data_file = raw_data_path(config, contract)
        if not data_file.exists():
            parser.error(f"brak pełnych danych {data_file} - fingerprint nie może być policzony")
    else:
        data_file = config.ci_sample_path

    metadata = build_metadata(
        model_name=args.model_name,
        model_format=model_format,
        features=config.model_features,
        target=contract.target,
        positive_class=contract.positive_class,
        metrics=record["models"][args.model_name],
        data_fingerprint=file_fingerprint(data_file),
    )
    package_dir = save_package(model, args.out, metadata)
    print(f"Pakiet zapisany: {package_dir}")

    if args.wandb_project:
        aliases = tuple(args.alias or ["staging"])
        name = log_package_artifact(
            package_dir, project=args.wandb_project, entity=args.wandb_entity, aliases=aliases
        )
        print(f"W&B Artifact: {name} (aliasy: {', '.join(aliases)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
