from dataclasses import dataclass
from typing import Optional
from mlflow_export_import.client import databricks_cli_utils
from mlflow_export_import.client.capabilities import configure_provider_logging
from mlflow_export_import.client.provider import TrackingProvider
from mlflow_export_import.common import MlflowExportImportException
from mlflow_export_import.common import utils

_logger = utils.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedTrackingEndpoint:
    provider: TrackingProvider
    tracking_uri: str
    host: Optional[str]
    token: Optional[str]


def get_mlflow_host():
    """ Returns the MLflow tracking URI (host) """
    return resolve_mlflow_tracking_endpoint().host


def get_mlflow_host_token():
    """ 
    Returns the MLflow tracking URI (host) and auth token.
    For Databricks, expects the MLflow tracking URI in the form of 'databricks' or 'databricks://MY_PROFILE'.
    """
    endpoint = resolve_mlflow_tracking_endpoint()
    return (endpoint.host, endpoint.token)


def resolve_mlflow_tracking_endpoint(tracking_uri=None):
    """
    Resolve MLFLOW_TRACKING_URI to an explicit provider and request-ready host/token pair.
    """
    import mlflow
    uri = tracking_uri or mlflow.tracking.get_tracking_uri()
    if not uri:
        _raise_exception(uri)

    if uri.startswith("http://") or uri.startswith("https://"):
        return ResolvedTrackingEndpoint(TrackingProvider.MLFLOW, uri, uri, None)

    if uri.startswith("azureml://"):
        try:
            configure_provider_logging(TrackingProvider.AZUREML)
            client = mlflow.MlflowClient()
            creds = client._tracking_client.store.get_host_creds()
            host = getattr(creds, "host", None)
            token = getattr(creds, "token", None)
            if host:
                return ResolvedTrackingEndpoint(TrackingProvider.AZUREML, uri, host, token)
            _logger.warning(f"Could not resolve host for Azure ML tracking URI '{uri}'")
            return ResolvedTrackingEndpoint(TrackingProvider.AZUREML, uri, None, None)
        except Exception as e:
            _logger.warning(e)
            return ResolvedTrackingEndpoint(TrackingProvider.AZUREML, uri, None, None)

    if not uri.startswith("databricks"):
        _raise_exception(uri)

    try:
        toks = uri.split("//")
        profile = uri.split("//")[1] if len(toks) > 1 else None
        host, token = databricks_cli_utils.get_host_token_for_profile(profile)
        return ResolvedTrackingEndpoint(TrackingProvider.DATABRICKS, uri, host, token)
    # databricks_cli.utils.InvalidConfigurationError 
    # requests.exceptions.InvalidSchema(f"No connection adapters were found for {url!r}")
    except Exception as e: 
        _logger.warning(e)
        return ResolvedTrackingEndpoint(TrackingProvider.DATABRICKS, uri, None, None)


def _raise_exception(uri):
    raise MlflowExportImportException(
      f"MLflow tracking URI (MLFLOW_TRACKING_URI environment variable) must be one of HTTP(S), Databricks, or AzureML URI formats: '{uri}'.",
      http_status_code=401)
