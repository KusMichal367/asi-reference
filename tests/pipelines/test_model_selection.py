"""Benchmark i wybór modelu (Sprint 3) — w CI bez W&B i bez treningu AutoGluona."""

import copy
from unittest.mock import patch

import pandas as pd
import pytest

from asi_project.benchmark import package_size_mb
from asi_project.experiment_tracking import log_model_selection_to_wandb
from asi_project.model_package import load_package, predict
from asi_project.pipeline_registry import register_pipelines
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)
from asi_project.pipelines.data_science.nodes import evaluate_models, split_data, train_models
from asi_project.pipelines.model_selection.nodes import (
    benchmark_models,
    build_comparison_table,
    select_production_model,
)

SPLIT = {"test_size": 0.2, "random_state": 42}
RULE = {
    "primary_metric": "f1",
    "metric_tolerance": 0.02,
    "limits": {
        "max_latency_single_p95_ms": 100,
        "max_artifact_size_mb": 100,
        "max_load_memory_mb": 1024,
    },
    "tie_breakers": ["latency_single_p95_ms", "artifact_size_mb", "load_memory_mb"],
}


def _row(name: str, f1: float, p95: float, size: float, memory: float, **extra) -> dict:
    return {
        "candidate": name,
        "family": "autogluon",
        "best_model": name,
        "selectable": True,
        "model_format": "autogluon",
        "source_model_path": f"data/06_models/autogluon/{name}",
        "package_dir": f"data/06_models/candidates/{name}",
        "f1": f1,
        "precision": f1,
        "recall": f1,
        "accuracy": 0.9,
        "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
        "train_time_s": 1.0,
        "latency_single_p50_ms": p95 / 2,
        "latency_single_p95_ms": p95,
        "batch_ms_per_row": 0.1,
        "artifact_size_mb": size,
        "load_memory_mb": memory,
        "load_time_s": 0.1,
        "package_reproduces_metrics": True,
        **extra,
    }


@pytest.fixture
def benchmark_record() -> dict:
    return {
        "dataset": "iranian_churn",
        "target": "Churn",
        "positive_class": "1",
        "data_mode": "sample",
        "data_fingerprint": "fp",
        "split": {"test_size": 0.2, "random_state": 42, "n_train": 80, "n_test": 20},
        "features": ["Age"],
        "measurement": {},
        "candidates": [
            _row("dummy", 0.0, 0.5, 0.01, 1.0, selectable=False),
            _row("baseline", 0.60, 5.0, 0.01, 10.0),
            _row("ensemble", 0.89, 30.0, 2.0, 50.0),
            _row("single_gbm", 0.88, 10.0, 0.6, 40.0),
        ],
    }


# -=-=-= reguła wyboru =-=-=-


def test_cheaper_model_within_tolerance_wins(benchmark_record):
    selection = select_production_model(benchmark_record, RULE)

    assert selection["selected"] == "single_gbm"
    assert set(selection["shortlist"]) == {"ensemble", "single_gbm"}
    assert "baseline" in selection["rejected"]
    assert "dummy" in selection["rejected"]


def test_zero_tolerance_picks_best_metric(benchmark_record):
    selection = select_production_model(benchmark_record, {**RULE, "metric_tolerance": 0.0})
    assert selection["selected"] == "ensemble"


def test_limits_reject_candidates(benchmark_record):
    rule = copy.deepcopy(RULE)
    rule["limits"]["max_artifact_size_mb"] = 0.5
    selection = select_production_model(benchmark_record, rule)

    assert selection["selected"] == "baseline"
    assert any("artifact_size_mb" in reason for reason in selection["rejected"]["ensemble"])


def test_package_not_reproducing_metrics_is_rejected(benchmark_record):
    benchmark_record["candidates"][3]["package_reproduces_metrics"] = False
    selection = select_production_model(benchmark_record, RULE)
    assert selection["selected"] == "ensemble"


def test_no_eligible_candidate_raises(benchmark_record):
    rule = copy.deepcopy(RULE)
    rule["limits"]["max_latency_single_p95_ms"] = 0.1
    with pytest.raises(ValueError, match="Żaden kandydat"):
        select_production_model(benchmark_record, rule)


def test_comparison_table_marks_selected(benchmark_record):
    selection = select_production_model(benchmark_record, RULE)
    table = build_comparison_table(benchmark_record, selection)

    assert table["selected"].sum() == 1
    assert table.loc[table["selected"], "candidate"].item() == "single_gbm"
    assert list(table["f1"]) == sorted(table["f1"], reverse=True)
    for column in ("f1", "latency_single_p95_ms", "artifact_size_mb", "load_memory_mb"):
        assert column in table.columns


# -=-=-= benchmark na prawdziwych pakietach (sklearn, próbka CI) =-=-=-


@pytest.fixture
def trained_on_sample(project):
    raw = pd.read_csv(project.config.ci_sample_path)
    prepared = filter_data(normalize_data(validate_data(raw)))
    X_train, X_test, y_train, y_test = split_data(prepared, SPLIT)
    model, dummy = train_models(X_train, y_train)
    metrics = evaluate_models(model, dummy, X_train, X_test, y_test, SPLIT, "sample")
    return model, dummy, metrics, X_test, y_test


def test_benchmark_packages_and_measures_in_fresh_process(trained_on_sample, tmp_path):
    model, dummy, metrics, X_test, y_test = trained_on_sample
    params = {"candidates_dir": str(tmp_path / "candidates"), "latency_repeats": 5}

    record = benchmark_models(model, dummy, metrics, {"experiments": {}}, X_test, y_test, params)

    rows = {row["candidate"]: row for row in record["candidates"]}
    assert set(rows) == {"dummy", "logistic_regression"}
    for row in rows.values():
        assert row["package_reproduces_metrics"], row["candidate"]
        assert row["artifact_size_mb"] == package_size_mb(row["package_dir"])
        assert row["latency_single_p95_ms"] >= row["latency_single_p50_ms"] > 0
        assert row["n_rows"] == len(X_test)

    # Pakiet kandydata to zwykły pakiet modelu: wczytuje go ta sama funkcja co smoke i API.
    loaded, metadata = load_package(rows["logistic_regression"]["package_dir"])
    assert metadata["model_name"] == "logistic_regression"
    assert len(predict(loaded, metadata, X_test.head(5))) == 5

    selection = select_production_model(record, {**RULE, "limits": {}})
    assert selection["selected"] == "logistic_regression"


def test_benchmark_refuses_different_splits(trained_on_sample, tmp_path):
    model, dummy, metrics, X_test, y_test = trained_on_sample
    autogluon_results = {
        "data_mode": "sample",
        "split": {**metrics["split"], "random_state": 7},
        "experiments": {"x": {}},
    }
    with pytest.raises(ValueError, match="random_state"):
        benchmark_models(
            model,
            dummy,
            metrics,
            autogluon_results,
            X_test,
            y_test,
            {"candidates_dir": str(tmp_path)},
        )


# -=-=-= Kedro i W&B =-=-=-


def test_model_selection_pipeline_is_opt_in():
    from kedro.framework.project import configure_project

    configure_project("asi_project")
    pipelines = register_pipelines()

    assert "model_selection" in pipelines
    names = {node.name for node in pipelines["model_selection"].nodes}
    assert {"benchmark_models", "select_production_model", "log_model_selection"} <= names
    default_names = {node.name for node in pipelines["__default__"].nodes}
    assert "benchmark_models" not in default_names


def test_log_model_selection_to_wandb_disabled(benchmark_record, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr("asi_project.experiment_tracking.get_git_sha", lambda root: "sha123")
    monkeypatch.setattr(
        "asi_project.experiment_tracking.get_data_fp", lambda dataset, mode, root: "fp456"
    )
    selection = select_production_model(benchmark_record, RULE)
    table = build_comparison_table(benchmark_record, selection)

    with patch("asi_project.experiment_tracking.wandb.init") as mock_init:
        run = mock_init.return_value.__enter__.return_value
        run.id, run.url, run.summary = "run-ms", None, {}
        result = log_model_selection_to_wandb(
            benchmark_record, selection, table, project="asi-test", root=tmp_path
        )

    kwargs = mock_init.call_args.kwargs
    assert result["run_id"] == "run-ms"
    assert kwargs["job_type"] == "model-selection"
    assert kwargs["config"]["git_sha"] == "sha123"
    assert kwargs["config"]["data_fingerprint"] == "fp456"
    assert run.summary["selected_model"] == "single_gbm"
    assert run.summary["single_gbm/latency_single_p95_ms"] == 10.0


def test_package_model_script_packages_selected_model(trained_on_sample, tmp_path, monkeypatch):
    import importlib.util
    import json
    import pickle

    from asi_project.pipelines.model_selection.nodes import _data_fingerprint

    model, _, metrics, X_test, _ = trained_on_sample
    model_path = tmp_path / "trained_model.pkl"
    model_path.write_bytes(pickle.dumps(model))
    selection = {
        "selected": "logistic_regression",
        "best_model": "LogisticRegression",
        "model_format": "sklearn",
        "source_model_path": None,
        "data_mode": "sample",
        "data_fingerprint": _data_fingerprint("sample"),
        "metrics": metrics["models"]["logistic_regression"],
        "rule": RULE,
        "deployment": {"latency_single_p95_ms": 1.0},
    }
    selection_path = tmp_path / "selected_model.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("package_model", "scripts/package_model.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    # Test nie może czytać lokalnego .env (klucz W&B) ani łączyć się z W&B.
    monkeypatch.setattr("asi_project.experiment_tracking.load_local_env", lambda *a, **k: None)
    out = tmp_path / "package"
    code = script.main(
        [
            "--selection",
            str(selection_path),
            "--model",
            str(model_path),
            "--out",
            str(out),
            "--wandb-project",
            "",
        ]
    )

    assert code == 0
    loaded, metadata = load_package(out)
    assert metadata["model_name"] == "logistic_regression"
    assert metadata["best_model"] == "LogisticRegression"
    assert metadata["selection"]["rule"]["primary_metric"] == "f1"
    assert metadata["metrics"]["f1"] == metrics["models"]["logistic_regression"]["f1"]
    assert len(predict(loaded, metadata, X_test.head(5))) == 5


def test_json_reports_are_ascii_safe_on_windows():
    """JSONDataset zapisuje w kodowaniu systemowym (cp1250 na Windows), a czyta jako UTF-8.
    Polskie znaki w raportach muszą więc trafiać do pliku jako escape (ensure_ascii=True)."""
    import yaml

    catalog = yaml.safe_load(open("conf/base/catalog.yml", encoding="utf-8"))
    for name, entry in catalog.items():
        if entry.get("type") == "json.JSONDataset":
            assert (entry.get("save_args") or {}).get("ensure_ascii", True), name
