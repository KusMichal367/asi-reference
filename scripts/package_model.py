"""Buduje pakiet modelu z wyników pipeline'u i (opcjonalnie) wysyła go do W&B."""

import argparse
import os
import pickle

from asi_project.config import get_contract, load_project_config
from asi_project.data import raw_data_path
from asi_project.metrics import load_metrics
from asi_project.model_package import (
    build_metadata,
    file_fingerprint,
    log_package_artifact,
    save_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/06_models/trained_model.pkl")
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

    with open(args.model, "rb") as handle:
        model = pickle.load(handle)
    record = load_metrics(args.metrics)

    data_file = raw_data_path(config, contract)
    if record.get("data_mode") == "sample" or not data_file.exists():
        data_file = config.ci_sample_path

    metadata = build_metadata(
        model_name=args.model_name,
        model_format="sklearn",
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
