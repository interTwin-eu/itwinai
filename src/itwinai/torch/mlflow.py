# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, List

import mlflow
import pandas as pd
import yaml
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from itwinai.constants import PROFILING_AVG_NAME

cli_logger = logging.getLogger("cli_logger")


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
    cli_logger.info(f"MLFlow's artifacts URI: {run.info.artifact_uri}")

    mlflow_conf["experiment_name"] = experiment_name
    mlflow_conf["run_id"] = mlflow.active_run().info.run_id

    tmp_dir_path = Path(tmp_dir)
    _mlflow_log_pl_config(pl_config, tmp_dir_path / "pl_config.yml")


def teardown_lightning_mlflow() -> None:
    """End active mlflow run, if any."""
    if mlflow.active_run() is not None:
        mlflow.end_run()


def get_epoch_time_runs_by_parent(
    mlflow_client: MlflowClient,
    experiment_id: str,
    run: Run,
) -> List[Run]:
    """Get all epoch time runs associated with a given run.
    This function assumes that the data is in the main worker run of each train run.
    Which is either:
    - The main worker run in each trial run of a given tuner run (if Ray was used)
    - The main worker run of the given training run (if Ray was not used)
    Args:
        mlflow_client (mlflow.tracking.MlflowClient): MLFlow client to use.
        experiment_id (str): The ID of the experiment to search in.
        run (mlflow.entities.Run): The run from which to collect epoch runs.
    Returns:
        List[Run]: A list of runs that contain epoch time data associated with the given run.
    """

    def _children(parent_run_id: str) -> List[Run]:
        return mlflow_client.search_runs(
            [experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )

    first_level_children = _children(run.info.run_id)
    epoch_time_runs: List[Run] = []

    if not first_level_children:
        cli_logger.warning(
            f"No child runs found for run ID {run.info.run_id} in experiment {experiment_id}."
        )
        return epoch_time_runs

    for child in first_level_children:
        second_level_children = _children(child.info.run_id)
        # exists for ray runs
        if second_level_children:
            for grand_child in second_level_children:
                if "epoch_time_s" in grand_child.data.metrics:
                    epoch_time_runs.append(grand_child)
        elif "epoch_time_s" in child.data.metrics:
            epoch_time_runs.append(child)

    if len(epoch_time_runs) > 1:
        cli_logger.warning(
            f"Multiple epoch times found for run ID {run.info.run_id} in experiment"
            f" {experiment_id}. This indicates Ray HPO was used with multiple trials."
            " Hyperparameters can have a significant impact on epoch time, so keep in mind"
            " that the averaged epoch time data may not be comparable with other runs."
        )
    return epoch_time_runs


def get_profiling_avg_by_parent(
    mlflow_client: MlflowClient,
    experiment_id: str,
    run: Run,
) -> List[pd.DataFrame]:
    """Get all worker profiling averages associated with a given run.
    This function assumes that the worker profiling averages are either:
    - Nested under the trial runs of a tuner run (if Ray was used)
    - Nested under the training run (if Ray was not used)

    Args:
        mlflow_client (mlflow.tracking.MlflowClient): MLFlow client to use
        experiment_id (str): The ID of the experiment to search in.
        run (mlflow.entities.Run): The run from which to collect worker profiling averages.

    Returns:
        List[pd.DataFrame]: A list of DataFrames containing the worker profiling averages
            associated with the given run. Each DataFrame corresponds to a worker run.
    """

    def _children(parent_run_id: str) -> List[Run]:
        return mlflow_client.search_runs(
            [experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )

    first_level_children = _children(run.info.run_id)
    worker_profiling_averages: List[pd.DataFrame] = []

    if not first_level_children:
        cli_logger.warning(
            f"No child runs found for run ID {run.info.run_id} in experiment {experiment_id}."
        )
        return worker_profiling_averages

    leaf_nodes = []
    for child in first_level_children:
        second_level_children = _children(child.info.run_id)
        leaf_nodes += second_level_children if second_level_children else [child]

    for child in leaf_nodes:
        # Retrieve CSV artifact and convert to DataFrame
        # Cause MLflow is stupid we have to manual check if the artifact exists
        artifacts = mlflow_client.list_artifacts(child.info.run_id, path=PROFILING_AVG_NAME)
        rel_path = str(Path(PROFILING_AVG_NAME, f"{PROFILING_AVG_NAME}.csv"))
        if rel_path not in [artifact.path for artifact in artifacts]:
            continue  # Skip if the profiling averages artifact does not exist

        # If tracking_uri is local, no download will happen, otherwise downloads to temp
        local_csv = mlflow_client.download_artifacts(child.info.run_id, rel_path)
        cli_logger.info(f"Downloading profiling averages from {local_csv}")
        worker_avg_df = pd.read_csv(local_csv)
        worker_profiling_averages.append(worker_avg_df)

    if len(worker_profiling_averages) == 0:
        cli_logger.warning(
            f"No profiling averages found for run ID {run.info.run_id} in experiment"
            f" {experiment_id}."
        )

    return worker_profiling_averages


def get_gpu_runs_by_parent(
    mlflow_client: MlflowClient,
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
        cli_logger.warning(
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


def get_runs_by_name(
    mlflow_client: MlflowClient,
    experiment_id: str,
    run_names: List[str] | None = None,
) -> List[Run]:
    """Get all runs in an experiment by their names.

    Args:
        mlflow_client (mlflow.tracking.MlflowClient): MLFlow client to use.
        experiment_id (str): The ID of the experiment to search in.
        run_names (List[str] | None): The names of the runs to retrieve. If None, all runs
            in the experiment will be retrieved.

    Returns:
        List[Run]: A list of runs that match the given names.
    """
    if not run_names:
        # get all run IDs from the experiment that are not-nested
        runs = mlflow_client.search_runs([experiment_id])
        runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
        return runs

    runs = []
    # get all runs in the experiment
    for run_name in run_names:
        runs += mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"run_name='{run_name}'",
        )
    return runs
