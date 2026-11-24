import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "precision", "recall", "f1")


def load_local_env(root: Path = PROJECT_ROOT) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


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


def log_autogluon_experiments_to_wandb(
    record: dict[str, Any],
    *,
    project: str,
    entity: str | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, str | None]]:
    dataset = str(record["dataset"])
    data_mode = str(record["data_mode"])
    split = record["split"]
    git_sha = get_git_sha(root)
    data_fingerprint = get_data_fp(dataset, data_mode, root)

    runs = {}

    for exp_name, exp in record["experiments"].items():
        params = exp["params"]
        configuration = {
            "experiment_name": exp_name,
            "model_family": "autogluon",
            "best_model": exp["best_model"],
            "presets": params["presets"],
            "time_limit": params["time_limit"],
            "eval_metric": params["eval_metric"],
            "included_model_types": params.get("included_model_types"),
            "dataset": dataset,
            "data_mode": data_mode,
            "target": record["target"],
            "positive_class": record["positive_class"],
            "features": record["features"],
            "test_size": split["test_size"],
            "random_state": split["random_state"],
            "n_train": split["n_train"],
            "n_test": split["n_test"],
            "git_sha": git_sha,
            "data_fingerprint": data_fingerprint,
        }

        logged_metrics = {
            f"autogluon/{metric}": float(exp["metrics"][metric]) for metric in METRICS
        }
        logged_metrics.update(
            {
                "train_time_s": float(exp["train_time_s"]),
                "predict_time_s": float(exp["predict_time_s"]),
                "predict_ms_per_row": float(exp["predict_ms_per_row"]),
                "artifact_size_mb": float(exp["artifact_size_mb"]),
                "load_memory_mb": float(exp["load_memory_mb"]),
            }
        )

        with wandb.init(
            project=project,
            entity=entity,
            job_type="autogluon",
            name=f"autogluon-{exp_name}-{data_mode}",
            config=configuration,
            mode=os.getenv("WANDB_MODE", "disabled"),
        ) as run:
            run.log(logged_metrics)
            runs[exp_name] = {
                "run_id": run.id,
                "run_url": run.url,
            }

    return runs
