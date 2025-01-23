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
from typing import Set, Dict

import pandas as pd

from itwinai.scalability_report.utils import check_contains_columns


def read_scalability_metrics_from_csv(
    data_dir: Path | str, expected_columns: Set
) -> pd.DataFrame:
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
        check_contains_columns(
            df=df, expected_columns=expected_columns, file_path=file_path
        )
        dataframes.append(df)

    return pd.concat(dataframes)

