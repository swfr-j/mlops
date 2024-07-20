from piplines.training_pipelines import training_pipeline

if __name__ == "__main__":
    # Define the data path
    data_path = "/Users/ms253/Desktop/mlops/data/olist_customers_dataset.csv"

    # Run the pipeline
    training_pipeline(data_path)
