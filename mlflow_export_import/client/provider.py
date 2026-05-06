from enum import Enum


class TrackingProvider(Enum):
    MLFLOW = "mlflow"
    DATABRICKS = "databricks"
    AZUREML = "azureml"
