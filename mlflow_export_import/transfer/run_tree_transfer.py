import os
import tempfile

from mlflow_export_import.common import utils
from mlflow_export_import.common.iterators import SearchRunsIterator
from mlflow_export_import.client.mlflow_auth_utils import resolve_mlflow_tracking_endpoint
from mlflow_export_import.client.provider import TrackingProvider
from mlflow_export_import.run.export_run import export_run
from mlflow_export_import.run.import_run import import_run

_logger = utils.getLogger(__name__)


def collect_run_tree(src_client, root_run_id):
    """
    Return source runs in deterministic parent-first order for root run and descendants.
    """
    root_run = src_client.get_run(root_run_id)
    experiment_id = root_run.info.experiment_id
    endpoint = resolve_mlflow_tracking_endpoint(getattr(src_client, "tracking_uri", None))

    if endpoint.provider in (TrackingProvider.DATABRICKS, TrackingProvider.AZUREML):
        descendants = _collect_with_root_tag(src_client, experiment_id, root_run_id)
        if descendants:
            return [root_run] + descendants

    # Generic fallback that works for OSS and providers with parentRunId tags.
    return _collect_with_parent_tag(src_client, root_run)


def transfer_run_tree(src_client, dst_client, root_run_id, dst_experiment_name):
    src_runs = collect_run_tree(src_client, root_run_id)
    run_ids_map = {}
    failed_run_ids = []
    root_dst_run = None

    with tempfile.TemporaryDirectory() as tmp_root:
        for run in src_runs:
            src_run_id = run.info.run_id
            try:
                run_dir = os.path.join(tmp_root, src_run_id)
                exp_run = export_run(
                    run_id=src_run_id,
                    output_dir=run_dir,
                    notebook_formats=["SOURCE"],
                    mlflow_client=src_client
                )
                if not exp_run:
                    failed_run_ids.append(src_run_id)
                    continue
                dst_run, src_parent_run_id = import_run(
                    input_dir=run_dir,
                    experiment_name=dst_experiment_name,
                    mlflow_client=dst_client
                )
                if src_run_id == root_run_id:
                    root_dst_run = dst_run
                run_ids_map[src_run_id] = {
                    "dst_run_id": dst_run.info.run_id,
                    "src_parent_run_id": src_parent_run_id
                }
            except Exception as e:
                failed_run_ids.append(src_run_id)
                _logger.error(f"Failed transferring run '{src_run_id}': {e}")

    utils.nested_tags(dst_client, run_ids_map)
    _logger.info(
        f"Run tree transfer summary: total={len(src_runs)} transferred={len(run_ids_map)} failed={len(failed_run_ids)}"
    )
    if failed_run_ids:
        _logger.warning(f"Failed source run IDs: {failed_run_ids}")
    return root_dst_run, run_ids_map, failed_run_ids


def _collect_with_root_tag(client, experiment_id, root_run_id):
    """
    Databricks/Azure helper path.
    """
    filter_str = f"tags.mlflow.rootRunId = '{root_run_id}'"
    runs = list(SearchRunsIterator(client, experiment_id, filter=filter_str))
    if not runs:
        return []
    by_id = {run.info.run_id: run for run in runs}
    children_by_parent = {}
    for run in runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(run.info.run_id)
    return _parent_first_order(by_id, children_by_parent, root_run_id)


def _collect_with_parent_tag(client, root_run):
    experiment_id = root_run.info.experiment_id
    # Azure MLflow does not support LIKE for tag filters; fetch runs and filter in Python.
    runs = list(SearchRunsIterator(client, experiment_id))
    runs = [run for run in runs if run.data.tags.get("mlflow.parentRunId")]
    by_id = {run.info.run_id: run for run in runs}
    children_by_parent = {}
    for run in runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(run.info.run_id)
    return [root_run] + _parent_first_order(by_id, children_by_parent, root_run.info.run_id)


def _parent_first_order(by_id, children_by_parent, root_id):
    ordered = []
    stack = list(reversed(sorted(children_by_parent.get(root_id, []))))
    seen = set()
    while stack:
        run_id = stack.pop()
        if run_id in seen:
            continue
        seen.add(run_id)
        run = by_id.get(run_id)
        if run:
            ordered.append(run)
            children = sorted(children_by_parent.get(run_id, []))
            stack.extend(reversed(children))
    return ordered
