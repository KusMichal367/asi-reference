from typing import Any

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from asi_project.metrics import validate_metrics_record
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)
from asi_project.pipelines.data_science.nodes import (
    evaluate_models,
    split_data,
    train_models,
)

SPLIT = {"test_size": 0.2, "random_state": 42}


@pytest.fixture
def prepared_data(project) -> pd.DataFrame:
    raw = pd.read_csv(project.config.ci_sample_path)
    return filter_data(normalize_data(validate_data(raw)))


@pytest.fixture
def split(prepared_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return split_data(prepared_data, SPLIT)


@pytest.fixture
def trained(split) -> tuple[Pipeline, DummyClassifier]:
    X_train, _, y_train, _ = split
    return train_models(X_train, y_train)

def test_split_data_structure(prepared_data: pd.DataFrame, split, project) -> None:
    X_train, X_test, y_train, y_test = split

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(prepared_data)
    assert len(X_test) / len(prepared_data) == pytest.approx(SPLIT["test_size"], abs=0.02)
    # do X trafiają wyłącznie zadeklarowane cechy modelu (ALLOWED FEATURES)
    assert list(X_train.columns) == list(project.config.model_features)
    assert project.contract.target not in X_train.columns
    # obie klasy w obu zbiorach
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}

def test_split_data_is_reproducible(prepared_data: pd.DataFrame) -> None:
    first = split_data(prepared_data, SPLIT)
    second = split_data(prepared_data, SPLIT)

    pd.testing.assert_index_equal(first[1].index, second[1].index)

def test_train_models_returns_fitted_models(split, trained) -> None:
    _, X_test, _, _ = split
    model, dummy = trained

    assert isinstance(model, Pipeline)
    assert isinstance(dummy, DummyClassifier)
    for fitted in (model, dummy):
        predictions = fitted.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions) <= {0, 1}

def test_evaluate_models_returns_valid_metrics(split, trained, project) -> None:
    X_train, X_test, _, y_test = split
    model, dummy = trained

    record: dict[str, Any] = evaluate_models(
        model, dummy, X_train, X_test, y_test, SPLIT, data_mode="sample"
    )

    assert validate_metrics_record(record, project.contract) == []
    assert record["data_mode"] == "sample"
    assert record["split"]["n_train"] == len(X_train)
    assert record["split"]["n_test"] == len(X_test)
    assert {"dummy", "logistic_regression"} <= set(record["models"])
    for metrics in record["models"].values():
        assert 0.0 <= metrics["f1"] <= 1.0