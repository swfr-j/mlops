import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Clean the data and divide it into train and test data

    Args:
        df: pd.DataFrame: the data to clean

    Returns:
        X_train: pd.DataFrame: the training data
        X_test: pd.DataFrame: the testing data
        y_train: pd.Series: the training labels
        y_test: pd.Series: the testing
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        data_divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, data_divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaned and divided")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        return e
