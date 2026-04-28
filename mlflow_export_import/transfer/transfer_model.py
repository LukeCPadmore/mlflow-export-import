import tempfile
import click

from mlflow_export_import.common import utils
from mlflow_export_import.copy import copy_utils
from mlflow_export_import.model.export_model import export_model
from mlflow_export_import.model.import_model import import_model

_logger = utils.getLogger(__name__)


def transfer_model(
        src_model,
        dst_model,
        dst_experiment_name,
        src_tracking_uri,
        dst_tracking_uri
    ):
    src_client = copy_utils.mk_client(src_tracking_uri)
    dst_client = copy_utils.mk_client(dst_tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ok, _ = export_model(
            model_name=src_model,
            output_dir=tmp_dir,
            mlflow_client=src_client
        )
        if not ok:
            raise click.ClickException(f"Model '{src_model}' could not be exported from source.")
        import_model(
            model_name=dst_model,
            experiment_name=dst_experiment_name,
            input_dir=tmp_dir,
            mlflow_client=dst_client
        )
        _logger.info(
            f"Transferred model '{src_model}' to destination model '{dst_model}' "
            f"using destination experiment '{dst_experiment_name}'."
        )


@click.command()
@click.option("--src-model", type=str, required=True, help="Source model name.")
@click.option("--dst-model", type=str, required=True, help="Destination model name.")
@click.option("--dst-experiment-name", type=str, required=True, help="Destination experiment for imported runs.")
@click.option("--src-tracking-uri", type=str, required=True, help="Source MLflow tracking URI.")
@click.option("--dst-tracking-uri", type=str, required=True, help="Destination MLflow tracking URI.")
def main(src_model, dst_model, dst_experiment_name, src_tracking_uri, dst_tracking_uri):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")
    transfer_model(src_model, dst_model, dst_experiment_name, src_tracking_uri, dst_tracking_uri)


if __name__ == "__main__":
    main()
