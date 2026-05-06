import click
import mlflow

from mlflow_export_import.common import utils
from mlflow_export_import.common.click_options import opt_run_id, opt_experiment_name
from mlflow_export_import.transfer.run_tree_transfer import transfer_run_tree
from . import copy_utils
from . click_options import opt_src_mlflow_uri, opt_dst_mlflow_uri

_logger = utils.getLogger(__name__)


def copy(
        src_run_id, 
        dst_experiment_name, 
        src_mlflow_uri = None, 
        dst_mlflow_uri = None
    ):
    """
    Copies a run to another tracking server (workspace).

    :param src_run_id: Source run ID.
    :param dst_experiment_name: Destination experiment name.
    :param : src_mlflow_uri: Source tracking server (workspace) URI.
    :param : dst_mlflow_uri: Destination tracking server (workspace) URI.

    :return: Destination Run object.
    """

    return _copy(src_run_id, dst_experiment_name, 
        copy_utils.mk_client(src_mlflow_uri), 
        copy_utils.mk_client(dst_mlflow_uri)
    )


def _copy(src_run_id, dst_experiment_name, src_client=None, dst_client=None):
    src_client = src_client or mlflow.MlflowClient()
    dst_client = dst_client or mlflow.MlflowClient()
    dst_run, _, failed_run_ids = transfer_run_tree(
        src_client, dst_client, src_run_id, dst_experiment_name, import_source_tags=True
    )
    if not dst_run:
        raise Exception(f"Root run '{src_run_id}' could not be copied.")
    if failed_run_ids:
        _logger.warning(f"Copy completed with failed nested runs: {failed_run_ids}")
    return dst_run


@click.command()
@opt_run_id
@opt_experiment_name
@opt_src_mlflow_uri
@opt_dst_mlflow_uri
def main(run_id, experiment_name, src_mlflow_uri, dst_mlflow_uri):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    copy(run_id, experiment_name, src_mlflow_uri, dst_mlflow_uri)


if __name__ == "__main__":
    main()
