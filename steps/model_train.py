import logging
import pandas as pd
from zenml import step


@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train a model on the ingested data

    Args:
        df: pd.DataFrame: the data to train on
    """
    pass
