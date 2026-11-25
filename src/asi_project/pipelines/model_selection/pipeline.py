"""Pipeline ``model_selection``: benchmark kandydatów → wybór modelu → tabela → W&B.

Wymaga wcześniejszego uruchomienia:
    kedro run                         # baseline: trained_model, dummy_model, metrics
    kedro run --pipeline autogluon    # autogluon_results + katalogi modeli
Następnie:
    kedro run --pipeline model_selection
"""

from kedro.pipeline import Pipeline, node, pipeline

from asi_project.pipelines.data_processing import (
    create_pipeline as create_data_processing_pipeline,
)
from asi_project.pipelines.data_science.nodes import split_data

from .nodes import (
    benchmark_models,
    build_comparison_table,
    log_model_selection,
    select_production_model,
)


def create_pipeline(**kwargs: object) -> Pipeline:
    return create_data_processing_pipeline() + pipeline(
        [
            node(
                func=split_data,
                inputs=["prepared_data", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=benchmark_models,
                inputs=[
                    "trained_model",
                    "dummy_model",
                    "metrics",
                    "autogluon_results",
                    "X_test",
                    "y_test",
                    "params:model_selection",
                ],
                outputs="model_benchmark",
                name="benchmark_models",
            ),
            node(
                func=select_production_model,
                inputs=["model_benchmark", "params:model_selection"],
                outputs="selected_model",
                name="select_production_model",
            ),
            node(
                func=build_comparison_table,
                inputs=["model_benchmark", "selected_model"],
                outputs="model_comparison",
                name="build_comparison_table",
            ),
            node(
                func=log_model_selection,
                inputs=["model_benchmark", "selected_model", "model_comparison"],
                outputs="model_selection_wandb_run",
                name="log_model_selection",
            ),
        ]
    )
