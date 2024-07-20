import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Class to ingest data from a given path
    """

    def __init__(self, data_path: str) -> None:
        """
        Args:
            data_path: str: path to
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the data from the data_path

        Returns:
            pd.DataFrame: the ingested data
        """
        logging.info("Ingesting data from {}".format(self.data_path))
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path

    Args:
        data_path: str: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        return None
