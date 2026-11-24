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


def _metadata(project, model_format: str) -> dict:
    config, contract = project.config, project.contract
    return build_metadata(
        model_name="test",
        model_format=model_format,
        features=list(config.model_features),
        target=contract.target,
        positive_class=contract.positive_class,
        metrics={"f1": 0.5},
        data_fingerprint=file_fingerprint(config.ci_sample_path),
        git_commit="test",
    )


@pytest.fixture
def trained(sample, project):
    config, contract = project.config, project.contract
    y = encode_target(sample[contract.target], contract.positive_class)
    model = build_baseline_pipeline(config.numeric_features, config.categorical_features)
    return model.fit(sample[list(config.model_features)], y)


@pytest.fixture
def package(tmp_path, trained, project):
    return save_package(trained, tmp_path / "package", _metadata(project, "sklearn"))


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


def test_predictions_match_model_before_save(package, trained, sample, project) -> None:
    model, metadata = load_package(package)
    X = sample[list(project.config.model_features)]

    result = predict(model, metadata, sample)

    assert (result["prediction"].to_numpy() == trained.predict(X)).all()


def test_autogluon_package_without_training_data(tmp_path, sample, project) -> None:
    tabular = pytest.importorskip("autogluon.tabular")
    config, contract = project.config, project.contract
    features = list(config.model_features)
    train = sample[features].assign(
        label=encode_target(sample[contract.target], contract.positive_class)
    )
    predictor = tabular.TabularPredictor(label="label", path=str(tmp_path / "ag"), verbosity=0).fit(
        train, hyperparameters={"GBM": {}}, time_limit=60
    )
    predictor.set_decision_threshold(0.8)
    expected = predictor.predict(sample[features]).to_numpy()

    package_dir = save_package(predictor, tmp_path / "package", _metadata(project, "autogluon"))
    data_files = {"X.pkl", "y.pkl", "X_val.pkl", "y_val.pkl"}
    assert not [path for path in package_dir.rglob("*.pkl") if path.name in data_files]

    model, metadata = load_package(package_dir)
    result = predict(model, metadata, sample)
    assert (result["prediction"].to_numpy() == expected).all()
