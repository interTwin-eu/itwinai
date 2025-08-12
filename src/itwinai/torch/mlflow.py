# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, List

import mlflow
import mlflow.tracking
import pandas as pd
import yaml
from mlflow.entities import Run

py_logger = logging.getLogger(__name__)


def _get_mlflow_logger_conf(pl_config: Dict) -> Dict | None:
    """Extract MLFLowLogger configuration from pytorch lightning
    configuration file, if present.

    Args:
        pl_config (Dict): lightning configuration loaded in memory.

    Returns:
        Optional[Dict]: if present, MLFLowLogger constructor arguments
        (under 'init_args' key).
    """
    if not pl_config["trainer"].get("logger"):
        return None
    if isinstance(pl_config["trainer"]["logger"], list):
        # If multiple loggers are provided
        for logger_conf in pl_config["trainer"]["logger"]:
            if logger_conf["class_path"].endswith("MLFlowLogger"):
                return logger_conf["init_args"]
    elif pl_config["trainer"]["logger"]["class_path"].endswith("MLFlowLogger"):
        return pl_config["trainer"]["logger"]["init_args"]


def _mlflow_log_pl_config(pl_config: Dict, local_yaml_path: str | Path) -> None:
    if isinstance(local_yaml_path, str):
        local_yaml_path = Path(local_yaml_path)

    local_yaml_path.parent.mkdir(exist_ok=True, parents=True)
    with open(local_yaml_path, "w") as outfile:
        yaml.dump(pl_config, outfile, default_flow_style=False)
    mlflow.log_artifact(str(local_yaml_path))


def init_lightning_mlflow(
    pl_config: Dict,
    default_experiment_name: str = "Default",
    tmp_dir: str = ".tmp",
    **autolog_kwargs,
) -> None:
    """Initialize mlflow for pytorch lightning, also setting up
    auto-logging (mlflow.pytorch.autolog(...)). Creates a new mlflow
    run and attaches it to the mlflow auto-logger.

    Args:
        pl_config (Dict): pytorch lightning configuration loaded in memory.
        default_experiment_name (str, optional): used as experiment name
            if it is not given in the lightning conf. Defaults to 'Default'.
        tmp_dir (str): where to temporarily store some artifacts.
        autolog_kwargs (kwargs): args for mlflow.pytorch.autolog(...).
    """
    mlflow_conf: Dict | None = _get_mlflow_logger_conf(pl_config)
    if not mlflow_conf:
        return

    tracking_uri = mlflow_conf.get("tracking_uri")
    if not tracking_uri:
        save_path = mlflow_conf.get("save_dir")
        tracking_uri = "file://" + str(Path(save_path).resolve())

    experiment_name = mlflow_conf.get("experiment_name")
    if not experiment_name:
        experiment_name = default_experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(**autolog_kwargs)
    run = mlflow.start_run()
    py_logger.info(f"MLFlow's artifacts URI: {run.info.artifact_uri}")

    mlflow_conf["experiment_name"] = experiment_name
    mlflow_conf["run_id"] = mlflow.active_run().info.run_id

    tmp_dir_path = Path(tmp_dir)
    _mlflow_log_pl_config(pl_config, tmp_dir_path / "pl_config.yml")


def teardown_lightning_mlflow() -> None:
    """End active mlflow run, if any."""
    if mlflow.active_run() is not None:
        mlflow.end_run()


def get_gpu_data_by_run(
    mlflow_client: mlflow.tracking.MlflowClient,
    experiment_id: str,
    run: Run,
) -> List[Run]:
    """Get all GPU worker runs associated with a given run.
    This function assumes that the GPU worker runs are either:
    - Nested under the trial runs of a tuner run (if Ray was used)
    - Nested under the training run (if Ray was not used)

    Args:
        mlflow_client (mlflow.tracking.MlflowClient): MLFlow client to use.
        experiment_id (str): The ID of the experiment to search in.
        run (mlflow.entities.Run): The run from which to collect GPU worker runs.

    Returns:
        List[Run]: A list of runs that are GPU workers associated with the given run.
    """

    def _children(parent_run_id: str) -> List[Run]:
        return mlflow_client.search_runs(
            [experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )

    first_level_children = _children(run.info.run_id)
    gpu_runs: List[Run] = []

    if not first_level_children:
        py_logger.warning(
            f"No child runs found for run ID {run.info.run_id} in experiment {experiment_id}."
        )
        return gpu_runs

    for child in first_level_children:
        second_level_children = _children(child.info.run_id)
        if second_level_children:
            for grand_child in second_level_children:
                if any("gpu" in metric for metric in grand_child.data.metrics):
                    gpu_runs.append(grand_child)
        else:
            if any("gpu" in metric for metric in child.data.metrics):
                gpu_runs.append(child)

    return gpu_runs


def get_metric_names(run: Run) -> List[str]:
    """Get the names of all metrics logged in a run."""
    run_data = run.data.to_dictionary()
    metric_names = list(run_data["metrics"].keys())
    return metric_names


def get_params(run: Run) -> Dict[str, str]:
    """Get the parameters logged in a run."""
    run_data = run.data.to_dictionary()
    params = run_data["params"]
    return params


def get_run_metrics_as_df(
    mlflow_client: mlflow.MlflowClient,
    run: Run,
    metric_names: List[str] | None = None,
):
    """Collect metrics logged in a run and return them as a tidy DataFrame.

    Args:
        mlflow_client (mlflow.MlflowClient): MLFlow client to use.
        run (mlflow.entities.Run): The run from which to collect metrics.
        metric_names (List[str] | None): If provided, only these metrics
            will be collected. If None, all metrics will be collected.
    Returns:
        pd.DataFrame: A DataFrame containing the metrics, with columns:
            - metric_name: the name of the metric
            - sample_idx: the step index of the metric
            - timestamp: the timestamp of the metric
            - value: the value of the metric
            - all parameters logged in the run
    """
    if metric_names is None:
        metric_names = get_metric_names(run)

    params = get_params(run)

    collected_metrics = []
    for metric_name in metric_names:
        metric_history = mlflow_client.get_metric_history(
            run_id=run.info.run_id, key=metric_name
        )
        pd_convertible_metric_history = [
            {
                "metric_name": metric.key,
                "sample_idx": int(metric.step),
                "timestamp": int(metric.timestamp),
                "value": metric.value,
                **params,
            }
            for metric in metric_history
        ]
        collected_metrics += pd_convertible_metric_history

    metrics_df = pd.DataFrame.from_records(collected_metrics)
    return metrics_df
