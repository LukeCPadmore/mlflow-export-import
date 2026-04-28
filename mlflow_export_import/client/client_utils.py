import mlflow
from . http_client import HttpClient, MlflowHttpClient, DatabricksHttpClient, AzureHttpClient
from . import mlflow_auth_utils
from .capabilities import configure_provider_logging
from .provider import TrackingProvider

_PROVIDER_TO_CLIENT_CLS = {
    TrackingProvider.MLFLOW: MlflowHttpClient,
    TrackingProvider.DATABRICKS: DatabricksHttpClient,
    TrackingProvider.AZUREML: AzureHttpClient,
}


def _create_provider_client(provider, host, token):
    client_cls = _PROVIDER_TO_CLIENT_CLS.get(provider, MlflowHttpClient)
    return client_cls(host, token)


def create_http_client(mlflow_client, model_name=None):
    """
    Create MLflow HTTP client from MlflowClient.
    If model_name is a Unity Catalog (UC) model, the returned client is UC-enabled.
    """
    from mlflow_export_import.common import model_utils
    creds = mlflow_client._tracking_client.store.get_host_creds()
    if model_name and model_utils.is_unity_catalog_model(model_name):
        return HttpClient("api/2.0/mlflow/unity-catalog", creds.host, creds.token)
    tracking_uri = getattr(mlflow_client, "tracking_uri", None)
    endpoint = mlflow_auth_utils.resolve_mlflow_tracking_endpoint(tracking_uri)
    return _create_provider_client(endpoint.provider, creds.host, creds.token)


def create_dbx_client(mlflow_client):
    """
    Create Databricks HTTP client from MlflowClient.
    Returns None if not using Databricks backend.
    """
    try:
        creds = mlflow_client._tracking_client.store.get_host_creds()
        return DatabricksHttpClient(creds.host, creds.token)
    except AttributeError:
        # FileStore or other non-Databricks backend - return None
        return None


def create_mlflow_client():
    """
    Create MLflowClient. If MLFLOW_TRACKING_URI is UC, then set MlflowClient.tracking_uri to the non-UC variant.
    """
    try:
        endpoint = mlflow_auth_utils.resolve_mlflow_tracking_endpoint()
        configure_provider_logging(endpoint.provider)
    except Exception:
        # Keep local/non-supported tracking URI behavior unchanged.
        pass
    registry_uri = mlflow.get_registry_uri()
    if registry_uri:
        tracking_uri = mlflow.get_tracking_uri()
        nonuc_tracking_uri = tracking_uri.replace("databricks-uc","databricks") # NOTE: legacy
        return mlflow.MlflowClient(nonuc_tracking_uri, registry_uri)
    else:
        return mlflow.MlflowClient()
