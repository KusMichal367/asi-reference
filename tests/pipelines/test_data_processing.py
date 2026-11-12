import pandas as pd
import pytest

from asi_project.data import DataContractError
from asi_project.pipelines.data_processing.nodes import (
    filter_data,
    normalize_data,
    validate_data,
)


@pytest.fixture
def raw_data(project) -> pd.DataFrame:
    return pd.read_csv(project.config.ci_sample_path)


@pytest.fixture
def prepared_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    return filter_data(normalize_data(validate_data(raw_data)))


def test_validate_data_accepts_valid_sample(raw_data: pd.DataFrame, project) -> None:
    result = validate_data(raw_data)

    assert len(result) == len(raw_data)
    assert project.contract.target in result.columns


def test_validate_data_normalizes_column_names(raw_data: pd.DataFrame) -> None:
    messy = raw_data.rename(columns={"Call Failure": "Call  Failure "})

    result = validate_data(messy)

    assert "Call Failure" in result.columns
    assert "Call  Failure " not in result.columns


def test_validate_data_rejects_missing_target(raw_data: pd.DataFrame, project) -> None:
    broken = raw_data.drop(columns=[project.contract.target])

    with pytest.raises(DataContractError):
        validate_data(broken)


def test_normalize_data_strips_text_and_keeps_rows() -> None:
    df = pd.DataFrame({"text": ["  a ", "b  ", None], "num": [1, 2, 3]})

    result = normalize_data(df)

    assert result["text"].tolist()[:2] == ["a", "b"]
    assert result["num"].tolist() == [1, 2, 3]
    assert len(result) == len(df)
    # node nie modyfikuje wejścia
    assert df["text"].iloc[0] == "  a "


def test_filter_data_removes_duplicates_and_missing_target(
    prepared_data: pd.DataFrame, project
) -> None:
    target = project.contract.target
    duplicate = prepared_data.iloc[[0]]
    no_target = prepared_data.iloc[[1]].assign(**{target: None})
    dirty = pd.concat([prepared_data, duplicate, no_target], ignore_index=True)

    result = filter_data(dirty)

    assert len(result) == len(prepared_data)
    assert result[target].notna().all()
    assert not result.duplicated().any()


def test_filter_data_only_removes_rows(prepared_data: pd.DataFrame, raw_data: pd.DataFrame) -> None:
    assert len(prepared_data) <= len(raw_data)
    assert list(prepared_data.columns) == list(validate_data(raw_data).columns)
