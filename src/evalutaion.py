import logging
from abc import ABC, abstractmethod
import numpy as np
from numpy.core.multiarray import array as array
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class for defining evaluation strategies for our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.array, y_pred: np.array) -> None:
        """
        Calculate scores for the model
        Args:
            y_true: np.array: the true labels
            y_pred: np.array: the predicted labels
        Returns:
            None
        """
        pass


class MSE(Evaluation):
    """
    Evaluation strategy for Mean Squared Error
    """

    def calculate_scores(self, y_true: np.array, y_pred: np.array) -> None:
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            return e


class R2(Evaluation):
    """
    Evaluation strategy for R2
    """

    def calculate_scores(self, y_true: np.array, y_pred: np.array) -> None:
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R2: {e}")
            return e


class RMSE(Evaluation):
    """
    Evaluation strategy for Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.array, y_pred: np.array) -> None:
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RMSE: {e}")
            return e
