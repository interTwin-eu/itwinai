# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import hour as SECONDS_IN_HOUR

matplotlib.use("Agg")


def calculate_average_gpu_utilization(gpu_data_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the average GPU utilization for each strategy and
    number of GPUs.

    Returns:
        pd.DataFrame: A DataFrame containing the average gpu utilization for
            each strategy and number of GPUs, with the columns ``strategy``,
            ``num_global_gpus`` and ``utilization``.
    """
    required_columns = {"strategy", "utilization", "num_global_gpus"}
    if not required_columns.issubset(set(gpu_data_df.columns)):
        missing_columns = set(required_columns) - set(gpu_data_df.columns)
        raise ValueError(
            f"DataFrame is missing the following columns: {missing_columns}"
        )

    utilization_data = []
    grouped_df = gpu_data_df.groupby(["strategy", "num_global_gpus"])
    for (strategy, num_gpus), group in grouped_df:
        utilization_data.append(
            {
                "strategy": strategy,
                "num_global_gpus": num_gpus,
                "utilization": group["utilization"].mean(),
            }
        )
    return pd.DataFrame(utilization_data)


def calculate_total_energy_expenditure(gpu_data_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the total energy expenditure in Watt hours for each strategy and
    number of GPUs. Expects that the existence of the appropriate DataFrame columns is
    handled before calling this function.

    Returns:
        pd.DataFrame: A DataFrame containing the total expenditure in Watt hours for
            each strategy and number of GPUs, with the columns ``strategy``,
            ``num_global_gpus`` and ``total_energy_wh``.
    """
    required_columns = {"strategy", "power", "num_global_gpus", "probing_interval"}
    if not required_columns.issubset(set(gpu_data_df.columns)):
        missing_columns = set(required_columns) - set(gpu_data_df.columns)
        raise ValueError(
            f"DataFrame is missing the following columns: {missing_columns}"
        )
    energy_data = []
    grouped_df = gpu_data_df.groupby(["strategy", "num_global_gpus"])
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


def gpu_bar_plot(
    data_df: pd.DataFrame, plot_title: str, y_label: str, main_column: str
) -> Tuple[Figure, Axes]:
    """Makes a bar plot of the given data for each strategy and number of GPUs.

    Args:
        data_df: The dataframe to extract the data from.
        plot_title: The title to give the plot.
        y_label: The label for the y-axis.
        main_column: The column to use for the height of the bar plot.
    """
    required_columns = {"strategy", "num_global_gpus", main_column}
    if not required_columns.issubset(set(data_df.columns)):
        missing_columns = set(required_columns) - set(data_df.columns)
        raise ValueError(
            f"DataFrame is missing the following columns: {missing_columns}"
        )

    sns.set_theme()

    strategies = data_df["strategy"].unique()
    unique_gpu_counts = np.array(data_df["num_global_gpus"].unique())

    fig, ax = plt.subplots()
    x = np.arange(len(unique_gpu_counts))

    bar_width = 1 / (len(strategies) + 1)
    static_offset = (len(strategies) - 1) / 2
    for strategy_idx, strategy in enumerate(strategies):
        dynamic_bar_offset = strategy_idx - static_offset
        strategy_data = data_df[data_df["strategy"] == strategy]

        # Ensuring the correct spacing of the bars
        strategy_num_gpus = len(strategy_data["num_global_gpus"])

        ax.bar(
            x=x[:strategy_num_gpus] + dynamic_bar_offset * bar_width,
            height=strategy_data[main_column],
            width=bar_width,
            label=strategy,
        )

    ax.set_xlabel("Num GPUs")
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_gpu_counts)
    ax.legend(title="Strategy")

    figure_width = max(int(2 * len(unique_gpu_counts)), 8)
    fig.set_figwidth(figure_width)
    fig.set_figheight(figure_width * 0.8)

    sns.reset_orig()

    return fig, ax
