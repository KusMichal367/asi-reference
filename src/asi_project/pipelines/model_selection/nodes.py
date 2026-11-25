"""Benchmark kandydatów i wybór modelu produkcyjnego (Sprint 3).

Wszyscy kandydaci (baseline, DummyClassifier jako punkt odniesienia, konfiguracje AutoGluona)
są zapisywani wspólną funkcją ``save_package`` i mierzeni w świeżym procesie
(``asi_project.benchmark``). Reguła wyboru jest jawna i sterowana parametrami
``params:model_selection`` — decyzja w ADR-2 wynika z tej reguły, a nie z ręcznego wyboru.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from asi_project.benchmark import measure_in_fresh_process
from asi_project.config import get_contract, load_project_config
from asi_project.data import raw_data_path
from asi_project.model_package import build_metadata, file_fingerprint, save_package

SPLIT_KEYS = ("test_size", "random_state", "n_train", "n_test")
COMPARISON_COLUMNS = (
    "candidate",
    "family",
    "best_model",
    "f1",
    "precision",
    "recall",
    "latency_single_p50_ms",
    "latency_single_p95_ms",
    "batch_ms_per_row",
    "artifact_size_mb",
    "load_memory_mb",
    "load_time_s",
    "train_time_s",
    "shortlisted",
    "selected",
)
DEFAULT_TIE_BREAKERS = ("latency_single_p95_ms", "artifact_size_mb", "load_memory_mb")
LIMIT_KEYS = {
    "max_latency_single_p95_ms": "latency_single_p95_ms",
    "max_artifact_size_mb": "artifact_size_mb",
    "max_load_memory_mb": "load_memory_mb",
}


def _check_same_split(metrics: dict[str, Any], autogluon_results: dict[str, Any]) -> None:
    """Porównanie ma sens tylko na tym samym zbiorze testowym."""
    if not autogluon_results.get("experiments"):
        return
    problems = [
        key
        for key in SPLIT_KEYS
        if metrics["split"].get(key) != autogluon_results["split"].get(key)
    ]
    if metrics.get("data_mode") != autogluon_results.get("data_mode"):
        problems.append("data_mode")
    if problems:
        raise ValueError(
            "Baseline i AutoGluon liczone na innym podziale danych "
            f"({', '.join(problems)}). Uruchom ponownie oba pipeline'y."
        )


def _data_fingerprint(data_mode: str) -> str:
    config = load_project_config()
    if data_mode == "full":
        return file_fingerprint(raw_data_path(config, get_contract(config)))
    return file_fingerprint(config.ci_sample_path)


def collect_candidates(
    trained_model: Any,
    dummy_model: Any,
    metrics: dict[str, Any],
    autogluon_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lista kandydatów: model w pamięci albo ścieżka do katalogu AutoGluona + jego metryki."""
    candidates = [
        {
            "candidate": "dummy",
            "family": "sklearn",
            "best_model": "DummyClassifier",
            "selectable": False,  # tylko punkt odniesienia
            "model": dummy_model,
            "model_format": "sklearn",
            "metrics": metrics["models"]["dummy"],
            "train_time_s": None,
        },
        {
            "candidate": "logistic_regression",
            "family": "sklearn",
            "best_model": "LogisticRegression",
            "selectable": True,
            "model": trained_model,
            "model_format": "sklearn",
            "metrics": metrics["models"]["logistic_regression"],
            "train_time_s": None,
        },
    ]
    for exp_name, exp in (autogluon_results.get("experiments") or {}).items():
        candidates.append(
            {
                "candidate": f"autogluon_{exp_name}",
                "family": "autogluon",
                "best_model": exp["best_model"],
                "selectable": True,
                "model_path": exp["model_path"],
                "model_format": "autogluon",
                "metrics": exp["metrics"],
                "train_time_s": exp.get("train_time_s"),
                "params": exp.get("params"),
            }
        )
    return candidates


def _load_candidate_model(candidate: dict[str, Any]) -> Any:
    if "model" in candidate:
        return candidate["model"]
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(candidate["model_path"])


def benchmark_models(
    trained_model: Any,
    dummy_model: Any,
    metrics: dict[str, Any],
    autogluon_results: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_selection: dict[str, Any],
) -> dict[str, Any]:
    """Pakuje każdego kandydata i mierzy go w świeżym procesie na tym samym zbiorze testowym."""
    _check_same_split(metrics, autogluon_results)
    config = load_project_config()
    contract = get_contract(config)
    data_mode = str(metrics["data_mode"])
    data_fingerprint = _data_fingerprint(data_mode)
    repeats = int(model_selection.get("latency_repeats", 100))

    candidates_dir = Path(model_selection.get("candidates_dir", "data/06_models/candidates"))
    candidates_dir.mkdir(parents=True, exist_ok=True)
    test_csv = candidates_dir / "_test_set.csv"
    X_test.assign(**{contract.target: y_test.values}).to_csv(test_csv, index=False)

    rows = []
    for candidate in collect_candidates(trained_model, dummy_model, metrics, autogluon_results):
        package_dir = candidates_dir / candidate["candidate"]
        if package_dir.exists():
            shutil.rmtree(package_dir)
        metadata = build_metadata(
            model_name=candidate["candidate"],
            model_format=candidate["model_format"],
            features=config.model_features,
            target=contract.target,
            positive_class=contract.positive_class,
            metrics=candidate["metrics"],
            data_fingerprint=data_fingerprint,
        )
        save_package(_load_candidate_model(candidate), package_dir, metadata)
        measured = measure_in_fresh_process(package_dir, test_csv, contract.target, repeats)

        f1 = float(candidate["metrics"]["f1"])
        rows.append(
            {
                "candidate": candidate["candidate"],
                "family": candidate["family"],
                "best_model": candidate["best_model"],
                "selectable": candidate["selectable"],
                "model_format": candidate["model_format"],
                "source_model_path": candidate.get("model_path"),
                "package_dir": package_dir.as_posix(),
                "params": candidate.get("params"),
                "f1": f1,
                "precision": float(candidate["metrics"]["precision"]),
                "recall": float(candidate["metrics"]["recall"]),
                "accuracy": float(candidate["metrics"]["accuracy"]),
                "confusion_matrix": candidate["metrics"].get("confusion_matrix"),
                "train_time_s": candidate["train_time_s"],
                **{key: measured[key] for key in measured if key not in ("model_name",)},
                # Pakiet po wczytaniu w nowym procesie musi dać ten sam wynik co w treningu.
                "package_reproduces_metrics": abs(measured["f1_from_package"] - f1) < 1e-6,
            }
        )

    return {
        "dataset": contract.dataset_id,
        "target": contract.target,
        "positive_class": contract.positive_class,
        "data_mode": data_mode,
        "data_fingerprint": data_fingerprint,
        "split": metrics["split"],
        "features": list(config.model_features),
        "measurement": {
            "where": "świeży proces Pythona, load_package + predict z model_package.py",
            "latency_repeats": repeats,
            "load_memory": "przyrost RSS po wczytaniu i pierwszej predykcji (psutil)",
            "artifact_size": "rozmiar katalogu pakietu (model + metadata.json)",
        },
        "candidates": rows,
    }


def select_production_model(
    model_benchmark: dict[str, Any], model_selection: dict[str, Any]
) -> dict[str, Any]:
    """Jawna reguła wyboru.

    1. Odrzucamy kandydatów niewybieralnych (Dummy), łamiących limity wdrożeniowe
       albo takich, których pakiet nie odtwarza metryk.
    2. Shortlista: główna metryka nie gorsza niż ``najlepsza - metric_tolerance``.
    3. Z shortlisty wygrywa kandydat najtańszy we wdrożeniu (kolejność ``tie_breakers``).
    """
    primary = str(model_selection.get("primary_metric", "f1"))
    tolerance = float(model_selection.get("metric_tolerance", 0.0))
    limits = model_selection.get("limits") or {}
    tie_breakers = tuple(model_selection.get("tie_breakers") or DEFAULT_TIE_BREAKERS)

    rejected: dict[str, list[str]] = {}
    eligible = []
    for row in model_benchmark["candidates"]:
        reasons = []
        if not row["selectable"]:
            reasons.append("punkt odniesienia, nie kandydat")
        if not row["package_reproduces_metrics"]:
            reasons.append("pakiet nie odtwarza metryk z treningu")
        for limit_key, column in LIMIT_KEYS.items():
            limit, value = limits.get(limit_key), row.get(column)
            if limit is not None and value is not None and value > float(limit):
                reasons.append(f"{column}={value} > {limit_key}={limit}")
        if reasons:
            rejected[row["candidate"]] = reasons
        else:
            eligible.append(row)

    if not eligible:
        raise ValueError(f"Żaden kandydat nie spełnia kryteriów: {rejected}")

    best_value = max(row[primary] for row in eligible)
    shortlist = [row for row in eligible if row[primary] >= best_value - tolerance]
    for row in eligible:
        if row not in shortlist:
            rejected[row["candidate"]] = [
                f"{primary}={row[primary]:.4f} < najlepsze {best_value:.4f} - {tolerance}"
            ]

    def cost(row: dict[str, Any]) -> tuple[float, ...]:
        return tuple(
            float("inf") if row.get(key) is None else float(row[key]) for key in tie_breakers
        )

    chosen = min(shortlist, key=cost)
    return {
        "selected": chosen["candidate"],
        "data_mode": model_benchmark["data_mode"],
        "data_fingerprint": model_benchmark["data_fingerprint"],
        "family": chosen["family"],
        "best_model": chosen["best_model"],
        "model_format": chosen["model_format"],
        "source_model_path": chosen["source_model_path"],
        "package_dir": chosen["package_dir"],
        "metrics": {key: chosen[key] for key in ("accuracy", "precision", "recall", "f1")}
        | {"confusion_matrix": chosen["confusion_matrix"]},
        "deployment": {
            key: chosen[key]
            for key in (
                "latency_single_p50_ms",
                "latency_single_p95_ms",
                "batch_ms_per_row",
                "artifact_size_mb",
                "load_memory_mb",
                "load_time_s",
            )
        },
        "rule": {
            "primary_metric": primary,
            "metric_tolerance": tolerance,
            "best_value": best_value,
            "limits": dict(limits),
            "tie_breakers": list(tie_breakers),
        },
        "shortlist": [row["candidate"] for row in shortlist],
        "rejected": rejected,
    }


def build_comparison_table(
    model_benchmark: dict[str, Any], selected_model: dict[str, Any]
) -> pd.DataFrame:
    """Tabela porównawcza (CSV w data/08_reporting, ten sam układ co w ADR-2)."""
    rows = []
    for row in model_benchmark["candidates"]:
        rows.append(
            {
                **{key: row.get(key) for key in COMPARISON_COLUMNS},
                "shortlisted": row["candidate"] in selected_model["shortlist"],
                "selected": row["candidate"] == selected_model["selected"],
            }
        )
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS)).sort_values(
        "f1", ascending=False, ignore_index=True
    )


def log_model_selection(
    model_benchmark: dict[str, Any],
    selected_model: dict[str, Any],
    model_comparison: pd.DataFrame,
) -> dict[str, str | None]:
    from asi_project.experiment_tracking import load_local_env, log_model_selection_to_wandb

    if "WANDB_MODE" not in os.environ:
        load_local_env()
    mode = os.getenv("WANDB_MODE", "disabled")
    project = os.getenv("WANDB_PROJECT")
    if not project:
        if mode != "disabled":
            raise ValueError("Nie ustawiono WANDB_PROJECT")
        project = "asi-churn-ci"

    return log_model_selection_to_wandb(
        model_benchmark,
        selected_model,
        model_comparison,
        project=project,
        entity=os.getenv("WANDB_ENTITY"),
    )
