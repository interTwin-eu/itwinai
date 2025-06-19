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
from typing import Dict

import mlflow
import yaml

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
