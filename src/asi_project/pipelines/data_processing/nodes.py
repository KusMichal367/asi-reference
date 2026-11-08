import pandas as pd

from asi_project.config import get_contract, load_project_config
from asi_project.data import (
    DataContractError,
    filter_training_rows,
    normalize_column_names,
    normalize_input,
    validate,
)


def validate_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    df = normalize_column_names(raw_data)
    problems = validate(df, contract, config)
    if problems:
        raise DataContractError("\n".join(problems))
    return df


def normalize_data(validated_data: pd.DataFrame) -> pd.DataFrame:
    return normalize_input(validated_data)


def filter_data(normalized_data: pd.DataFrame) -> pd.DataFrame:
    config = load_project_config()
    contract = get_contract(config)

    return filter_training_rows(normalized_data, contract)
