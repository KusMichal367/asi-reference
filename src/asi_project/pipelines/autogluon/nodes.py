import os
import shutil
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.metrics import compute_metrics


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 4)


def _measure_load_memory_mb(predictor_cls: Any, model_dir: Path) -> float:
    tracemalloc.start()
    try:
        predictor_cls.load(str(model_dir))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return round(peak / (1024 * 1024), 4)


def run_autogluon_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    split: dict[str, Any],
    data_mode: str,
    autogluon: dict[str, Any],
) -> dict[str, Any]:
    from autogluon.tabular import TabularPredictor

    config = load_project_config()
    contract = get_contract(config)

    experiments_cfg = autogluon.get("experiments") or {}
    if not experiments_cfg:
        raise ValueError("Brak konfiguracji eksperymentów w params:autogluon.experiments")

    eval_metric = str(autogluon.get("eval_metric", "f1"))
    models_root = Path(autogluon.get("models_dir", "data/06_models/autogluon"))
    train_data = X_train.assign(**{contract.target: y_train.values})

    experiments_results = {}

    for exp_name, exp_cfg in experiments_cfg.items():
        time_limit = exp_cfg.get("time_limit")
        if not isinstance(time_limit, int | float) or time_limit <= 0:
            raise ValueError(f"Eksperyment '{exp_name}' wymaga dodatniego time_limit.")

        presets = str(exp_cfg.get("presets", "medium_quality"))
        verbosity = int(exp_cfg.get("verbosity", 1))
        included_model_types = exp_cfg.get("included_model_types")

        model_dir = models_root / exp_name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.parent.mkdir(parents=True, exist_ok=True)

        fit_kwargs = {
            "presets": presets,
            "time_limit": float(time_limit),
        }
        if included_model_types:
            fit_kwargs["included_model_types"] = list(included_model_types)

        t0 = time.perf_counter()
        predictor = TabularPredictor(
            label=contract.target,
            eval_metric=eval_metric,
            path=str(model_dir),
            verbosity=verbosity,
        ).fit(train_data, **fit_kwargs)
        train_time_s = round(time.perf_counter() - t0, 4)

        t0 = time.perf_counter()
        y_pred = predictor.predict(X_test)
        predict_time_s = round(time.perf_counter() - t0, 6)
        predict_ms_per_row = round((predict_time_s / max(len(X_test), 1)) * 1000, 4)

        experiments_results[exp_name] = {
            "params": {
                "presets": presets,
                "time_limit": float(time_limit),
                "eval_metric": eval_metric,
                "included_model_types": list(included_model_types)
                if included_model_types
                else None,
            },
            "metrics": compute_metrics(y_test, y_pred),
            "best_model": str(predictor.model_best),
            "model_path": str(model_dir.as_posix()),
            "train_time_s": train_time_s,
            "predict_time_s": predict_time_s,
            "predict_ms_per_row": predict_ms_per_row,
            "artifact_size_mb": _dir_size_mb(model_dir),
            "load_memory_mb": _measure_load_memory_mb(TabularPredictor, model_dir),
        }

    return {
        "dataset": contract.dataset_id,
        "target": contract.target,
        "positive_class": contract.positive_class,
        "data_mode": data_mode,
        "eval_metric": eval_metric,
        "split": {
            "strategy": contract.split_strategy,
            "test_size": float(split["test_size"]),
            "random_state": int(split["random_state"]),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "features": list(config.model_features),
        "experiments": experiments_results,
    }


def log_autogluon_runs(autogluon_results: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    from asi_project.experiment_tracking import (
        load_local_env,
        log_autogluon_experiments_to_wandb,
    )

    if "WANDB_MODE" not in os.environ:
        load_local_env()

    mode = os.getenv("WANDB_MODE", "disabled")
    project = os.getenv("WANDB_PROJECT")

    if not project:
        if mode != "disabled":
            raise ValueError("Nie ustawiono WANDB_PROJECT")
        project = "asi-churn-ci"

    return log_autogluon_experiments_to_wandb(
        autogluon_results,
        project=project,
        entity=os.getenv("WANDB_ENTITY"),
    )
