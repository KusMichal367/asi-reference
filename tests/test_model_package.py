"""Test pakietu modelu w CI: zapis → odczyt → predykcja, bez W&B i bez pełnego treningu."""

import json

import pandas as pd
import pytest

from asi_project.data import encode_target, filter_training_rows, normalize_input
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import (
    METADATA_FILE,
    REQUIRED_KEYS,
    ModelPackageError,
    build_metadata,
    file_fingerprint,
    load_package,
    predict,
    save_package,
)


@pytest.fixture
def sample(project) -> pd.DataFrame:
    raw = pd.read_csv(project.config.ci_sample_path)
    return filter_training_rows(normalize_input(raw), project.contract)


@pytest.fixture
def package(tmp_path, sample, project):
    config, contract = project.config, project.contract
    features = list(config.model_features)
    y = encode_target(sample[contract.target], contract.positive_class)

    model = build_baseline_pipeline(config.numeric_features, config.categorical_features)
    model.fit(sample[features], y)

    metadata = build_metadata(
        model_name="logistic_regression",
        model_format="sklearn",
        features=features,
        target=contract.target,
        positive_class=contract.positive_class,
        metrics={"f1": 0.5},
        data_fingerprint=file_fingerprint(config.ci_sample_path),
        git_commit="test",
    )
    return save_package(model, tmp_path / "package", metadata)


def test_package_has_model_and_metadata(package) -> None:
    assert (package / METADATA_FILE).exists()
    assert (package / "model.joblib").exists()

    metadata = json.loads((package / METADATA_FILE).read_text(encoding="utf-8"))
    for key in REQUIRED_KEYS:
        assert key in metadata
    assert metadata["positive_class"] == "1"
    assert "scikit-learn" in metadata["library_versions"]


def test_load_and_predict_five_rows(package, sample) -> None:
    model, metadata = load_package(package)

    result = predict(model, metadata, sample.head(5))

    assert list(result.columns) == ["prediction", "probability"]
    assert len(result) == 5
    assert set(result["prediction"]) <= {0, 1}
    assert result["probability"].between(0, 1).all()


def test_predict_rejects_missing_features(package, sample) -> None:
    model, metadata = load_package(package)
    broken = sample.drop(columns=[metadata["features"][0]])

    with pytest.raises(ModelPackageError):
        predict(model, metadata, broken)


def test_load_rejects_missing_metadata(tmp_path) -> None:
    with pytest.raises(ModelPackageError):
        load_package(tmp_path)
