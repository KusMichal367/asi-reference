import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "precision", "recall", "f1")


def get_git_sha(root: Path = PROJECT_ROOT) -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    return res.stdout.strip()


def get_data_fp(dataset: str, data_mode: str, root: Path = PROJECT_ROOT) -> str:
    if data_mode == "full":
        path = root / "data" / "01_raw" / dataset / "fingerprint.json"
        fingerprint = json.loads(path.read_text(encoding="utf-8"))
        return str(fingerprint["sha256"])

    if data_mode == "sample":
        path = root / "data" / "01_raw" / "sample.csv"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    raise ValueError(f"Nieznany tryb danych:{data_mode}")


def log_metrics_to_wandb(
    record: dict[str, Any],
    *,
    project: str,
    entity: str | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, str | None]:

    dataset = str(record["dataset"])
    data_mode = str(record["data_mode"])
    split = record["split"]

    configuration = {
        "dataset": dataset,
        "data_mode": data_mode,
        "target": record["target"],
        "positive_class": record["positive_class"],
        "features": record["features"],
        "test_size": split["test_size"],
        "random_state": split["random_state"],
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "git_sha": get_git_sha(root),
        "data_fingerprint": get_data_fp(dataset, data_mode, root),
    }

    logged_metrics = {
        f"{model_name}/{metric}": float(values[metric])
        for model_name, values in record["models"].items()
        for metric in METRICS
    }

    with wandb.init(
        project=project,
        entity=entity,
        job_type="baseline",
        name=f"baseline-{data_mode}",
        config=configuration,
        mode=os.getenv("WANDB_MODE", "disabled"),
    ) as run:
        run.log(logged_metrics)
        return {
            "run_id": run.id,
            "run_url": run.url,
        }
