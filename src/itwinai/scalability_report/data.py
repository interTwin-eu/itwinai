# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import List, Set

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from itwinai.scalability_report.utils import check_contains_columns
from itwinai.torch.mlflow import (
    get_epoch_time_runs_by_parent,
    get_gpu_runs_by_parent,
    get_profiling_avg_by_parent,
    get_run_metrics_as_df,
    get_runs_by_name,
)

py_logger = logging.getLogger(__name__)


def read_profiling_data_from_mlflow(
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None = None,
    expected_columns: Set[str] | None = None,
) -> pd.DataFrame | None:
    """Reads and validates profiling data from a mlflow experiment and combines them into a
    single DataFrame.

    Args:
        mlflow_client (MlflowClient): An instance of MlflowClient to interact with MLflow.
        experiment_name (str): Name of the MLflow experiment to read from.
        run_names (List[str] | None): Name of the runs to read metrics from. If empty, all runs
        in the experiment will be considered.
        expected_columns (Set[str] | None): A set of column names expected to be present in
        the profiling data. If None, no validation is performed on the columns.

    Returns:
        pd.DataFrame | None: A DataFrame containing the concatenated data from all valid CSV
        files in the directory.
    """
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
    if experiment is None:
        py_logger.warning(
            f"Experiment '{experiment_name}' does not exist in MLflow at path"
            f" '{mlflow_client.tracking_uri}'."
        )
        return None

    runs = get_runs_by_name(
        mlflow_client,
        experiment.experiment_id,
        run_names=run_names,
    )

    profiling_dataframes = []
    for run in runs:
        profiling_avg_dataframes = get_profiling_avg_by_parent(
            mlflow_client, experiment.experiment_id, run
        )
        profiling_dataframes.extend(profiling_avg_dataframes)

    if not profiling_dataframes:
        # Warnings for no profiling data are already emitted in get_profiling_avg_by_parent
        return None

    profiling_dataframe_concat = pd.concat(profiling_dataframes)
    if expected_columns is not None:
        check_contains_columns(
            df=profiling_dataframe_concat,
            expected_columns=expected_columns,
        )

    return profiling_dataframe_concat


def read_epoch_time_from_mlflow(
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None = None,
) -> pd.DataFrame | None:
    """Reads and validates epoch time metrics from a mlflow experiment and combines them into a
    single DataFrame.

    Args:
        experiment_name (str): Name of the MLflow experiment to read from.
        run_names (List[str] | None): Name of the runs to read metrics from. If empty, all runs
        in the experiment will be considered.
    Returns:
        pd.DataFrame | None: A DataFrame containing the concatenated data from all epoch time
        metrics in the given runs of the experiment.
    """
    experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
    if experiment is None:
        py_logger.warning(
            f"Experiment '{experiment_name}' does not exist in MLflow at path '{mlflow_path}'."
        )
        return None

    runs = get_runs_by_name(
        mlflow_client,
        experiment.experiment_id,
        run_names=run_names,
    )

    epoch_time_dataframes = []
    for run in runs:
        epoch_time_runs = get_epoch_time_runs_by_parent(
            mlflow_client, experiment.experiment_id, run
        )
        for epoch_time_run in epoch_time_runs:
            epoch_time_dataframes.append(
                get_run_metrics_as_df(
                    mlflow_client,
                    epoch_time_run,
                    metric_names=["epoch_time_s"],
                )
            )

    if not epoch_time_dataframes:
        py_logger.warning(
            f"No epoch time found for experiment '{experiment_name}' with runs: {run_names}."
        )
        return None

    return pd.concat(epoch_time_dataframes)


def read_gpu_metrics_from_mlflow(
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None = None,
) -> pd.DataFrame | None:
    """Reads and validates GPU metrics from a mlflow experiment and combines them into a
    single DataFrame.

    Args:
        experiment_name (str): Name of the MLflow experiment to read from.
        run_names (List[str] | None): Name of the runs to read metrics from. If empty, all runs
        in the experiment will be considered.

    Returns:
        pd.DataFrame | None: A DataFrame containing the concatenated data from all gpu metrics
        in the given runs of the experiment.
    """
    mlflow.set_tracking_uri(mlflow_client.tracking_uri)

    experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
    if experiment is None:
        py_logger.warning(
            f"Experiment '{experiment_name}' does not exist in MLflow at path '{mlflow_path}'."
        )
        return None

    runs = get_runs_by_name(
        mlflow_client,
        experiment.experiment_id,
        run_names=run_names,
    )

    gpu_dataframes = []
    for run in runs:
        gpu_runs = get_gpu_runs_by_parent(mlflow_client, experiment.experiment_id, run)
        for gpu_run in gpu_runs:
            gpu_dataframes.append(
                get_run_metrics_as_df(
                    mlflow_client,
                    gpu_run,
                    metric_names=["gpu_utilization_percent", "gpu_power_W"],
                )
            )

    if not gpu_dataframes:
        py_logger.warning(
            f"No GPU metrics found for experiment '{experiment_name}' with runs: {run_names}."
        )
        return None

    return pd.concat(gpu_dataframes)


def read_scalability_metrics_from_csv(
    data_dir: Path | str, expected_columns: Set
) -> pd.DataFrame:
    """Reads and validates scalability metric CSV files from a directory and combines
    them into a single DataFrame.

    Args:
        data_dir (Path | str): Path to the directory containing the CSV files. All files
            in the directory must have a .csv extension.
        expected_columns (Set): A set of column names expected to be present in each CSV
            file.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all valid CSV
        files in the directory.

    Raises:
        ValueError: If the directory contains non-CSV files, if no .csv files are found,
        or if any file is missing the expected columns.
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    file_paths = list(data_dir.iterdir())

    # Checking that all files end with .csv
    if len([f for f in file_paths if f.suffix != ".csv"]) > 0:
        raise ValueError(
            f"Directory '{data_dir.resolve()} contains files with suffix different from .csv!"
        )

    if len(file_paths) == 0:
        raise ValueError(f"Found no .csv files in directory: '{data_dir.resolve()}'")

    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        check_contains_columns(df=df, expected_columns=expected_columns, file_path=file_path)
        dataframes.append(df)

    return pd.concat(dataframes)
