# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from os.path import isdir
from pathlib import Path
from typing import Set

import mlflow
import pandas as pd

from itwinai.scalability_report.utils import check_contains_columns
from itwinai.utils import RELATIVE_MLFLOW_PATH


# def read_gpu_metrics_from_mlflow(
#     experiment_name: Path | str,
#     run_name: str,
# ) -> pd.DataFrame:
#     """Reads and validates GPU metrics from a mlflow experiment and combines them into a
#     single DataFrame.
#
#     Args:
#         data_dir (Path | str): Path to the directory containing the CSV files. All files
#             in the directory must have a .csv extension.
#         expected_columns (Set): A set of column names expected to be present in each CSV
#             file.
#
#     Returns:
#         pd.DataFrame: A DataFrame containing the concatenated data from all valid CSV
#         files in the directory.
#
#     Raises:
#         ValueError: If the directory contains non-CSV files, if no .csv files are found,
#         or if any file is missing the expected columns.
#     """
#     mlflow_path = RELATIVE_MLFLOW_PATH.resolve()
#     if not isdir(mlflow_path):
#         raise ValueError(
#             f"Directory '{mlflow_path}' does not exist or is not a "
#             "directory."
#         )
#     mlflow.set_tracking_uri(RELATIVE_MLFLOW_PATH.resolve())
#
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment_by_name(experiment_name)
#
#     if experiment is None:
#         raise ValueError(f"Experiment '{experiment_name}' does not exist in MLflow at path '{mlflow_path}'.")
#
#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         filter_string=f"tags.mlflow.runName = '{run_name}'",
#     )
# )

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
            f"Directory '{data_dir.resolve()} contains files with suffix different "
            f"from .csv!"
        )

    if len(file_paths) == 0:
        raise ValueError(f"Found no .csv files in directory: '{data_dir.resolve()}'")

    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        check_contains_columns(df=df, expected_columns=expected_columns, file_path=file_path)
        dataframes.append(df)

    return pd.concat(dataframes)
