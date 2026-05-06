import tempfile
import click

from mlflow_export_import.common import utils
from mlflow_export_import.copy import copy_utils
from mlflow_export_import.experiment.export_experiment import export_experiment
from mlflow_export_import.experiment.import_experiment import import_experiment

_logger = utils.getLogger(__name__)


def transfer_experiment(
        experiment,
        dst_experiment_name,
        src_tracking_uri,
        dst_tracking_uri
    ):
    src_client = copy_utils.mk_client(src_tracking_uri)
    dst_client = copy_utils.mk_client(dst_tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        export_experiment(
            experiment_id_or_name=experiment,
            output_dir=tmp_dir,
            mlflow_client=src_client
        )
        run_info_map = import_experiment(
            experiment_name=dst_experiment_name,
            input_dir=tmp_dir,
            mlflow_client=dst_client
        )
        _logger.info(
            f"Transferred experiment '{experiment}' into destination experiment '{dst_experiment_name}' "
            f"with {len(run_info_map)} imported runs."
        )
        return run_info_map


@click.command()
@click.option("--experiment", type=str, required=True, help="Source experiment name or ID.")
@click.option("--dst-experiment-name", type=str, required=True, help="Destination experiment name.")
@click.option("--src-tracking-uri", type=str, required=True, help="Source MLflow tracking URI.")
@click.option("--dst-tracking-uri", type=str, required=True, help="Destination MLflow tracking URI.")
def main(experiment, dst_experiment_name, src_tracking_uri, dst_tracking_uri):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")
    transfer_experiment(experiment, dst_experiment_name, src_tracking_uri, dst_tracking_uri)


if __name__ == "__main__":
    main()
