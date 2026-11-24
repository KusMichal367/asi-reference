from kedro.pipeline import Pipeline, node, pipeline

from asi_project.pipelines.data_processing import (
    create_pipeline as create_data_processing_pipeline,
)
from asi_project.pipelines.data_science.nodes import split_data

from .nodes import log_autogluon_runs, run_autogluon_experiments


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
                func=run_autogluon_experiments,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:split",
                    "params:data_mode",
                    "params:autogluon",
                ],
                outputs="autogluon_results",
                name="run_autogluon_experiments",
            ),
            node(
                func=log_autogluon_runs,
                inputs="autogluon_results",
                outputs="autogluon_wandb_runs",
                name="log_autogluon_runs",
            ),
        ]
    )
