from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from asi_project.experiment_tracking import log_autogluon_experiments_to_wandb
from asi_project.pipeline_registry import register_pipelines
from asi_project.pipelines.autogluon.nodes import (
    log_autogluon_runs,
    run_autogluon_experiments,
)
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)
from asi_project.pipelines.data_science.nodes import split_data

SPLIT = {"test_size": 0.2, "random_state": 42}


class _FakePredictor:
    def __init__(self, label: str, eval_metric: str, path: str, verbosity: int = 1) -> None:
        self.label = label
        self.eval_metric = eval_metric
        self.path = Path(path)
        self.verbosity = verbosity
        self.model_best = "LightGBM"
        self.fit_kwargs = {}

    def fit(self, train_data: pd.DataFrame, **kwargs: Any) -> "_FakePredictor":
        self.fit_kwargs = kwargs
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "predictor.pkl").write_bytes(b"fake-model-bytes" * 64)
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series([i % 2 for i in range(len(data))], index=data.index)

    @classmethod
    def load(cls, path: str) -> "_FakePredictor":
        return cls(label="Churn", eval_metric="f1", path=path)


@pytest.fixture
def split(project):
    raw = pd.read_csv(project.config.ci_sample_path)
    prepared = filter_data(normalize_data(validate_data(raw)))
    return split_data(prepared, SPLIT)


def test_autogluon_pipeline_registered_and_excluded_from_default():
    from kedro.framework.project import configure_project

    configure_project("asi_project")
    pipelines = register_pipelines()

    assert "autogluon" in pipelines
    autogluon_nodes = {node.name for node in pipelines["autogluon"].nodes}
    default_nodes = {node.name for node in pipelines["__default__"].nodes}

    assert {"split_data", "run_autogluon_experiments", "log_autogluon_runs"} <= autogluon_nodes
    assert "run_autogluon_experiments" not in default_nodes
    assert "log_autogluon_runs" not in default_nodes


def test_run_autogluon_experiments_with_mocked_predictor(split, tmp_path, project, monkeypatch):
    X_train, X_test, y_train, y_test = split
    monkeypatch.setattr("autogluon.tabular.TabularPredictor", _FakePredictor)

    autogluon_cfg = {
        "eval_metric": "f1",
        "models_dir": str(tmp_path / "autogluon"),
        "experiments": {
            "fast_30s": {"presets": "medium_quality", "time_limit": 30},
            "quality_60s": {"presets": "good_quality", "time_limit": 60},
        },
    }

    result = run_autogluon_experiments(
        X_train,
        X_test,
        y_train,
        y_test,
        SPLIT,
        data_mode="sample",
        autogluon=autogluon_cfg,
    )

    assert result["dataset"] == project.contract.dataset_id
    assert result["data_mode"] == "sample"
    assert set(result["experiments"]) == {"fast_30s", "quality_60s"}
    for exp_name, exp in result["experiments"].items():
        assert exp["best_model"] == "LightGBM"
        assert 0.0 <= exp["metrics"]["f1"] <= 1.0
        assert exp["train_time_s"] >= 0.0
        assert exp["predict_time_s"] >= 0.0
        assert exp["artifact_size_mb"] > 0.0
        assert exp["load_memory_mb"] >= 0.0
        assert (tmp_path / "autogluon" / exp_name / "predictor.pkl").exists()


def test_run_autogluon_experiments_requires_positive_time_limit(split, tmp_path):
    X_train, X_test, y_train, y_test = split
    bad_cfg = {
        "eval_metric": "f1",
        "models_dir": str(tmp_path / "autogluon"),
        "experiments": {"no_limit": {"presets": "medium_quality", "time_limit": 0}},
    }

    with pytest.raises(ValueError, match="time_limit"):
        run_autogluon_experiments(
            X_train,
            X_test,
            y_train,
            y_test,
            SPLIT,
            data_mode="sample",
            autogluon=bad_cfg,
        )


def test_log_autogluon_experiments_to_wandb_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr("asi_project.experiment_tracking.get_git_sha", lambda root: "sha123")
    monkeypatch.setattr(
        "asi_project.experiment_tracking.get_data_fp",
        lambda dataset, data_mode, root: "fp456",
    )

    record = {
        "dataset": "iranian_churn",
        "data_mode": "sample",
        "target": "Churn",
        "positive_class": "1",
        "features": ["Age"],
        "split": {"test_size": 0.2, "random_state": 42, "n_train": 80, "n_test": 20},
        "experiments": {
            "fast_30s": {
                "params": {"presets": "medium_quality", "time_limit": 30.0, "eval_metric": "f1"},
                "metrics": {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75},
                "best_model": "LightGBM",
                "train_time_s": 5.0,
                "predict_time_s": 0.02,
                "predict_ms_per_row": 1.0,
                "artifact_size_mb": 2.5,
                "load_memory_mb": 12.0,
            },
            "quality_60s": {
                "params": {"presets": "good_quality", "time_limit": 60.0, "eval_metric": "f1"},
                "metrics": {"accuracy": 0.92, "precision": 0.85, "recall": 0.75, "f1": 0.8},
                "best_model": "WeightedEnsemble_L2",
                "train_time_s": 12.0,
                "predict_time_s": 0.05,
                "predict_ms_per_row": 2.5,
                "artifact_size_mb": 8.0,
                "load_memory_mb": 24.0,
            },
        },
    }

    with patch("asi_project.experiment_tracking.wandb.init") as mock_init:
        mock_run = mock_init.return_value.__enter__.return_value
        mock_run.id = "run-ag"
        mock_run.url = None

        runs = log_autogluon_experiments_to_wandb(record, project="asi-test", root=tmp_path)

        assert set(runs) == {"fast_30s", "quality_60s"}
        assert mock_init.call_count == 2
        first_kwargs = mock_init.call_args_list[0].kwargs
        assert first_kwargs["mode"] == "disabled"
        assert first_kwargs["job_type"] == "autogluon"
        assert first_kwargs["config"]["git_sha"] == "sha123"
        assert first_kwargs["config"]["data_fingerprint"] == "fp456"
        assert first_kwargs["config"]["presets"] == "medium_quality"
        assert first_kwargs["config"]["time_limit"] == 30.0


def test_log_autogluon_runs_requires_project_when_online(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    with pytest.raises(ValueError, match="WANDB_PROJECT"):
        log_autogluon_runs({"experiments": {}})
