from pathlib import Path
from re import Match, Pattern, compile
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import hour as SECONDS_IN_HOUR

matplotlib.use("Agg")


def read_energy_df(pattern: Optional[str], log_dir: Path) -> pd.DataFrame:
    """Read files matching the given regex pattern from directory and converting them
    into a Pandas DataFrame. If pattern is None, we assume a match on all files.
    Expects that the existence of ``log_dir`` is handled before calling this function.

    Args:
        pattern: The regex string used to match files.
        log_dir: The directory to search for files in.

    Raises:
        ValueError: If no matching files are found in the given logging directory.
    """

    pattern_re: Optional[Pattern] = None
    if pattern is not None:
        pattern_re = compile(pattern)

    # Load and concatenate dataframes
    dataframes = []
    for entry in log_dir.iterdir():
        match: Union[bool, Match] = True
        if pattern_re is not None:
            match = pattern_re.search(str(entry))

        if not match:
            continue

        print(f"Loading data from file: '{entry}' when creating energy DataFrame")
        df = pd.read_csv(entry)
        dataframes.append(df)

    if len(dataframes) == 0:
        if pattern is None:
            error_message = f"Unable to find any files in {log_dir.resolve()}!"
        else:
            error_message = (
                f"No files matched pattern, '{pattern}', in log_dir, "
                f"{log_dir.resolve()}!"
            )
        raise ValueError(error_message)

    return pd.concat(dataframes)


def calculate_aggregated_energy_expenditure(
    gpu_utilization_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculates the total energy expenditure in Watt hours for each strategy and
    number of GPUs. Expects that the existence of the appropriate DataFrame columns is
    handled before calling this function.

    Returns:
        pd.DataFrame: A DataFrame containing the total expenditure in Watt hours for
            each strategy and number of GPUs, with the columns ``strategy``,
            ``num_global_gpus`` and ``total_energy_wh``.
    """
    energy_data = []

    grouped_df = gpu_utilization_df.groupby(["strategy", "num_global_gpus"])
    for (strategy, num_gpus), group in grouped_df:

        if len(group["probing_interval"].unique()) != 1:
            raise ValueError(
                f"probing_interval must have the same value for each strategy and "
                f"number of GPUs, but was heterogeneous for strategy: {strategy} "
                f"and number of GPUs: {num_gpus}."
            )

        probing_interval = group["probing_interval"].iloc[0]
        total_energy_wh = group["power"].sum() * probing_interval / SECONDS_IN_HOUR
        energy_data.append(
            {
                "strategy": strategy,
                "num_global_gpus": num_gpus,
                "total_energy_wh": total_energy_wh,
            }
        )
    return pd.DataFrame(energy_data)


def gpu_energy_plot(gpu_utilization_df: pd.DataFrame) -> Tuple[Figure, Axes]:
    """Makes an energy bar plot of the GPU utilization dataframe, showing the total
    energy expenditure for each strategy and number of GPUs in Watt hours.
    """
    required_columns = {"strategy", "power", "num_global_gpus", "probing_interval"}
    if not required_columns.issubset(set(gpu_utilization_df.columns)):
        missing_columns = set(required_columns) - set(set(gpu_utilization_df.columns))
        raise ValueError(
            f"DataFrame is missing the following columns: {missing_columns}"
        )
    sns.set_theme()

    energy_df = calculate_aggregated_energy_expenditure(gpu_utilization_df)

    strategies = energy_df["strategy"].unique()
    unique_gpu_counts = np.array(energy_df["num_global_gpus"].unique())

    fig, ax = plt.subplots()
    x = np.arange(len(unique_gpu_counts))

    bar_width = 1 / (len(strategies) + 1)
    static_offset = (len(strategies) - 1) / 2
    for strategy_idx, strategy in enumerate(strategies):
        dynamic_bar_offset = strategy_idx - static_offset
        strategy_data = energy_df[energy_df["strategy"] == strategy]

        # Ensuring the correct spacing of the bars
        strategy_num_gpus = len(strategy_data["num_global_gpus"])

        ax.bar(
            x=x[:strategy_num_gpus] + dynamic_bar_offset * bar_width,
            height=strategy_data["total_energy_wh"],
            width=bar_width,
            label=strategy,
        )

    ax.set_xlabel("Num GPUs")
    ax.set_ylabel("Energy Consumption (Wh)")
    ax.set_title("Energy Consumption by Strategy and Number of GPUs")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_gpu_counts)
    ax.legend(title="Strategy")

    return fig, ax

def backup_data(file_paths: List): 
    pass
