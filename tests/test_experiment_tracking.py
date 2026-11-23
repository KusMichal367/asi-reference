import hashlib
import json
from unittest.mock import patch

import pytest

from asi_project.experiment_tracking import get_data_fp, log_metrics_to_wandb


def test_sample_fp(tmp_path):
    sample_path = tmp_path / "data/01_raw/sample.csv"
    sample_path.parent.mkdir(parents=True)
    content = b"Age,Churn\n25,0\n40,1\n"
    sample_path.write_bytes(content)

    res = get_data_fp("iranian_churn", "sample", tmp_path)

    assert res == hashlib.sha256(content).hexdigest()


def test_full_fp(tmp_path):
    fp_path = tmp_path / "data/01_raw/iranian_churn/fingerprint.json"
    fp_path.parent.mkdir(parents=True)
    fp_path.write_text(
        json.dumps({"sha256": "test-sha256"}),
        encoding="utf-8",
    )

    res = get_data_fp("iranian_churn", "full", tmp_path)

    assert res == "test-sha256"


def test_invalid_data_mode(tmp_path):
    with pytest.raises(ValueError, match="Nieznany tryb danych"):
        get_data_fp("iranian_churn", "unknown", tmp_path)


def test_log_metrics_to_wandb_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    monkeypatch.setattr(
        "asi_project.experiment_tracking.get_git_sha",
        lambda root: "abc123",
    )
    monkeypatch.setattr(
        "asi_project.experiment_tracking.get_data_fp",
        lambda dataset, data_mode, root: "fingerprint123",
    )

    record = {
        "dataset": "iranian_churn",
        "data_mode": "full",
        "target": "Churn",
        "positive_class": 1,
        "features": ["Age"],
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "n_train": 2280,
            "n_test": 570,
        },
        "models": {
            "logistic_regression": {
                "accuracy": 0.9018,
                "precision": 0.8235,
                "recall": 0.4719,
                "f1": 0.6,
            }
        },
    }

    with patch("asi_project.experiment_tracking.wandb.init") as mock_init:
        mock_run = mock_init.return_value.__enter__.return_value
        mock_run.id = "test-run"
        mock_run.url = None

        result = log_metrics_to_wandb(
            record,
            project="asi-test",
            root=tmp_path,
        )

        assert result["run_id"] == "test-run"

        kwargs = mock_init.call_args.kwargs
        assert kwargs["mode"] == "disabled"
        assert kwargs["config"]["git_sha"] == "abc123"
        assert kwargs["config"]["data_fingerprint"] == "fingerprint123"
        assert kwargs["config"]["random_state"] == 42

        mock_run.log.assert_called_once_with(
            {
                "logistic_regression/accuracy": 0.9018,
                "logistic_regression/precision": 0.8235,
                "logistic_regression/recall": 0.4719,
                "logistic_regression/f1": 0.6,
            }
        )
