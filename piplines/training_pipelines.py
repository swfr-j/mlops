from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline
def training_pipeline(data_path: str):
    """
    Executes the training pipeline for machine learning models.

    Args:
        data_path (str): The path to the data file.

    Returns:
        None
    """
    df = ingest_data(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)
