from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    """
    Executes the training pipeline for machine learning models.

    Args:
        data_path (str): The path to the data file.

    Returns:
        None
    """
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, y_train)
    r2, mse, rmse = evaluate_model(model, X_test, y_test)
