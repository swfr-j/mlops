import logging
import pandas as pd
from zenml import step
from src.evalutaion import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated


@step
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
        r2 = R2().calculate_scores(y_test, predictions)
        rmse = RMSE().calculate_scores(y_test, predictions)

        return r2, mse, rmse
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        return e
