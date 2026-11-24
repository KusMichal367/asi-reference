"""Rejestrowanie pipeline'ów Kedro (Sprint 2).

Po przygotowaniu ``create_pipeline()`` w podpakietach
``data_processing`` oraz ``data_science`` Kedro wykrywa je automatycznie.
Ten plik NIE zawiera implementacji nodów ani gotowego pipeline'u.
"""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Wykryj pipeline'y z ``src/asi_project/pipelines/``."""
    pipelines = find_pipelines()
    default_names = [name for name in pipelines if name != "autogluon"]
    pipelines["__default__"] = sum((pipelines[name] for name in default_names), Pipeline([]))
    return pipelines
