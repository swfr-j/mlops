from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """
    Parameters for the model name
    """

    model_name: str = "LinearRegression"
