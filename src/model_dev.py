import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        Args:
            X_train: pd.DataFrame: the training data
            y_train: pd.Series: the training labels
        Returns:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the model
        Args:
            X_train: pd.DataFrame: the training data
            y_train: pd.Series: the training labels
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained")
            return reg
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            return e
