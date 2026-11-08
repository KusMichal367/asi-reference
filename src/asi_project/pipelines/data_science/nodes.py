from dataclasses import replace
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import build_metrics_record, compute_metrics, validate_metrics_record
from asi_project.model import build_baseline_pipeline, make_dummy


def split_data(
    prepared_data: pd.DataFrame, split: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    config = load_project_config()
    contract = get_contract(config)

    X, y = make_xy(prepared_data, contract, config.model_features)

    return split_train_test(
        X, y, contract, test_size=split["test_size"], random_state=split["random_state"]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, DummyClassifier]:
    config = load_project_config()

    model = build_baseline_pipeline(
        config.numeric_features,
        config.categorical_features,
    )

    model.fit(X_train, y_train)

    dummy = make_dummy()
    dummy.fit(X_train, y_train)

    return model, dummy


def evaluate_models(
    model: Pipeline,
    dummy: DummyClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    split: dict[str, Any],
    data_mode: str,
) -> dict[str, Any]:

    config = replace(
        load_project_config(),
        test_size=float(split["test_size"]),
        random_state=int(split["random_state"]),
    )

    contract = get_contract(config)

    results = {
        "dummy": compute_metrics(y_test, dummy.predict(X_test)),
        "logistic_regression": compute_metrics(y_test, model.predict(X_test)),
    }

    record = build_metrics_record(
        contract=contract,
        config=config,
        features=config.model_features,
        n_train=len(X_train),
        n_test=len(X_test),
        data_mode=data_mode,
        results=results,
    )

    problems = validate_metrics_record(record, contract)
    if problems:
        raise ValueError("problem z metrykami: " + "; ".join(problems))

    return record
