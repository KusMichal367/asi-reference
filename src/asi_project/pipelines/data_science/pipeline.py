from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_models, split_data, train_models


def create_pipeline(**kwargs: object) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["prepared_data", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=train_models,
                inputs=["X_train", "y_train"],
                outputs=["trained_model", "dummy_model"],
                name="train_models",
            ),
            node(
                func=evaluate_models,
                inputs=[
                    "trained_model",
                    "dummy_model",
                    "X_train",
                    "X_test",
                    "y_test",
                    "params:split",
                    "params:data_mode",
                ],
                outputs="metrics",
                name="evaluate_models",
            ),
        ]
    )
