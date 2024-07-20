import logging
import mlflow
import pandas as pd
from zenml import step
from src.evalutaion import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "mse"],
    Annotated[float, "rmse"],
]:
    """
    Evaluate the model
    """
    try:
        predictions = model.predict(X_test)

        mse = MSE().calculate_scores(y_test, predictions)
        mlflow.log_metric("MSE", mse)

        r2 = R2().calculate_scores(y_test, predictions)
        mlflow.log_metric("R2", r2)

        rmse = RMSE().calculate_scores(y_test, predictions)
        mlflow.log_metric("RMSE", rmse)

        return r2, mse, rmse
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        return e
