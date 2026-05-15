import click
import mlflow

from mlflow_export_import.common import utils
from mlflow_export_import.copy import copy_utils
from mlflow_export_import.transfer.run_tree_transfer import transfer_run_tree

_logger = utils.getLogger(__name__)


def transfer_run(
        run_id,
        experiment_name,
        src_tracking_uri,
        dst_tracking_uri
    ):
    src_client = copy_utils.mk_client(src_tracking_uri)
    dst_client = copy_utils.mk_client(dst_tracking_uri)

    original_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(src_tracking_uri)
        dst_run, _, failed_run_ids = transfer_run_tree(
            src_client, dst_client, run_id, experiment_name, import_source_tags=True
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)

    if not dst_run:
        raise click.ClickException(f"Root run '{run_id}' could not be transferred.")
    if failed_run_ids:
        _logger.warning(f"Transfer completed with failed nested runs: {failed_run_ids}")
    _logger.info(f"Transferred run tree rooted at '{run_id}' to destination root run '{dst_run.info.run_id}'.")
    return dst_run


@click.command()
@click.option("--run-id", type=str, required=True, help="Source run ID.")
@click.option("--experiment-name", type=str, required=True, help="Destination experiment name.")
@click.option("--src-tracking-uri", type=str, required=True, help="Source MLflow tracking URI.")
@click.option("--dst-tracking-uri", type=str, required=True, help="Destination MLflow tracking URI.")
def main(run_id, experiment_name, src_tracking_uri, dst_tracking_uri):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")
    transfer_run(run_id, experiment_name, src_tracking_uri, dst_tracking_uri)


if __name__ == "__main__":
    main()
