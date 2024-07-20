import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a model on the ingested data

    Args:
        X_train: pd.DataFrame: the training data
        y_train: pd.Series: the training labels

    Returns:
        RegressorMixin: the trained model
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model not found")
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        return e
