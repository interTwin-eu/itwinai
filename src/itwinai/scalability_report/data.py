# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import uuid
from pathlib import Path
from typing import Set

import pandas as pd

from itwinai.scalability_report.utils import check_contains_columns


def read_epoch_time_data(
    epoch_time_dir: Path | str, expected_columns: Set
) -> pd.DataFrame:
    if isinstance(epoch_time_dir, str):
        epoch_time_dir = Path(epoch_time_dir)

    file_paths = list(epoch_time_dir.iterdir())

    # Checking that all files end with .csv
    if len([f for f in file_paths if f.suffix != ".csv"]) > 0:
        raise ValueError(
            f"Directory '{epoch_time_dir.resolve()} contains files with suffix different "
            f"from .csv!"
        )

    if len(file_paths) == 0:
        raise ValueError(
            f"Found no .csv files in directory: '{epoch_time_dir.resolve()}'"
        )

    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        check_contains_columns(
            df=df, expected_columns=expected_columns, file_path=file_path
        )
        dataframes.append(df)

    return pd.concat(dataframes)


def backup_scalability_metrics(
    metric_df: pd.DataFrame,
    experiment_name: str | None,
    run_name: str | None,
    backup_dir: str,
    filename: str,
) -> None:
    """Stores the data in the given dataframe as a .csv file in its own folder for the
    experiment name and its own subfolder for the run_name. If these are not provided,
    then they will be generated randomly using uuid4.
    """
    if experiment_name is None:
        random_id = str(uuid.uuid4())
        experiment_name = "exp_" + random_id[:6]
    if run_name is None:
        random_id = str(uuid.uuid4())
        run_name = "run_" + random_id[:6]

    backup_path = Path(backup_dir) / experiment_name / run_name / filename
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(backup_path, index=False)
    print(f"Storing backup file at '{backup_path.resolve()}'.")
