from dataclasses import dataclass
import logging

from .provider import TrackingProvider


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_logged_models_api: bool
    supports_traces_api: bool
    supports_artifact_download_strict: bool


_CAPABILITIES = {
    TrackingProvider.MLFLOW: ProviderCapabilities(
        supports_logged_models_api=True,
        supports_traces_api=True,
        supports_artifact_download_strict=True,
    ),
    TrackingProvider.DATABRICKS: ProviderCapabilities(
        supports_logged_models_api=True,
        supports_traces_api=True,
        supports_artifact_download_strict=True,
    ),
    TrackingProvider.AZUREML: ProviderCapabilities(
        supports_logged_models_api=False,
        supports_traces_api=False,
        supports_artifact_download_strict=False,
    ),
}

_AZURE_LOGGING_CONFIGURED = False


def get_provider_capabilities(provider):
    return _CAPABILITIES.get(provider, _CAPABILITIES[TrackingProvider.MLFLOW])


def configure_provider_logging(provider):
    global _AZURE_LOGGING_CONFIGURED
    if provider != TrackingProvider.AZUREML or _AZURE_LOGGING_CONFIGURED:
        return
    for logger_name in ("azure", "azure.identity", "azure.core", "azureml"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    _AZURE_LOGGING_CONFIGURED = True
