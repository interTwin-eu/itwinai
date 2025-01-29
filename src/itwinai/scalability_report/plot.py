# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from itertools import cycle
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator, ScalarFormatter

# Doing this because otherwise I get an error about X11 Forwarding which I believe
# is due to the server trying to pass the image to the client computer
matplotlib.use("Agg")

marker_cycle = cycle("ov^s*dXpD.+12348")


def calculate_plot_dimensions(num_datapoints: int) -> Tuple[int, int]:
    """Calculates the height and width of a plot, given a number of datapoints.

    Returns:
        int: The calculated height
        int: The calculated width
    """

    # Note: These calculations are somewhat arbitrary, so this is open to suggestions
    width = max(int(2 * num_datapoints), 8)
    height = min(int(width * 0.8), 6)
    return height, width


def absolute_avg_epoch_time_plot(
    avg_epoch_time_df: pd.DataFrame, gpus_per_node: int = 4
) -> Tuple[Figure, Axes]:
    """Generates a log-log plot of average epoch training times against the number of GPUs
    for distributed training strategies.

    Args:
        avg_epoch_time_df (pd.DataFrame): A DataFrame containing the following columns:
            - "nodes": Number of nodes used in the training process.
            - "avg_epoch_time": Average time (in seconds) taken for an epoch.
            - "name": Name of the distributed training strategy.
        gpus_per_node (int): Number of GPUs per node. Used to calculate the total number
            of GPUs for each training configuration. Defaults to 4.

    Returns:
        Tuple[Figure, Axes]: A tuple containing the matplotlib `Figure` and `Axes` objects
        of the generated plot.

    Raises:
        ValueError: If `avg_epoch_time_df` is missing required columns.
    """
    sns.set_theme()
    fig, ax = plt.subplots()

    unique_nodes = list(avg_epoch_time_df["nodes"].unique())
    unique_names = avg_epoch_time_df["name"].unique()
    num_gpus = [num_node * gpus_per_node for num_node in unique_nodes]
    for name in unique_names:
        data = avg_epoch_time_df[avg_epoch_time_df["name"] == name]

        marker = next(marker_cycle)
        ax.plot(
            data["nodes"] * gpus_per_node,
            data["avg_epoch_time"],
            marker=marker,
            label=name,
            linestyle="-",
            markersize=6,
        )

    # The log scale must be set before changing the ticks
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Calculating upper and lower bounds to get the appropriate top and bottom ticks
    upper_bound = 10 ** int(
        np.ceil(np.log10(avg_epoch_time_df["avg_epoch_time"].max()))
    )
    lower_bound = 10 ** int(
        np.floor(np.log10(avg_epoch_time_df["avg_epoch_time"].min()))
    )
    ax.set_ylim(lower_bound, upper_bound)

    # Remove minor ticks on x-axis and format new ticks
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(num_gpus)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Average Epoch Time (s)")
    ax.set_title("Average Epoch Time vs Number of GPUs")
    ax.legend(title="Method")

    num_datapoints = len(unique_nodes)
    figure_height, figure_width = calculate_plot_dimensions(
        num_datapoints=num_datapoints
    )
    fig.set_figheight(figure_height)
    fig.set_figwidth(figure_width)

    plt.tight_layout()
    sns.reset_orig()

    return fig, ax


def relative_epoch_time_speedup_plot(
    avg_epoch_time_df: pd.DataFrame, gpus_per_node: int = 4
) -> Tuple[Figure, Axes]:
    """Creates a log-log plot showing the relative training speedup for distributed
    training strategies as the number of GPUs increases.

    Args:
        avg_epoch_time_df (pd.DataFrame): A DataFrame containing the following columns:
            - "nodes": Number of nodes used in the training process.
            - "avg_epoch_time": Average time (in seconds) taken for an epoch.
            - "name": Name of the distributed training strategy.
        gpus_per_node (int): Number of GPUs per node. Used to calculate the total number
            of GPUs for each training configuration. Defaults to 4.

    Returns:
        Tuple[Figure, Axes]: A tuple containing the matplotlib `Figure` and `Axes` objects
        of the generated plot.

    Raises:
        ValueError: If `avg_epoch_time_df` is missing required columns.
    """
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(6, 4))

    avg_epoch_time_df["num_gpus"] = avg_epoch_time_df["nodes"] * gpus_per_node
    avg_epoch_time_df["linear_speedup"] = avg_epoch_time_df["nodes"].astype(float)

    # Plotting the speedup for each strategy
    strategy_names = sorted(avg_epoch_time_df["name"].unique())
    for strategy in strategy_names:
        strategy_data = avg_epoch_time_df[avg_epoch_time_df.name == strategy]

        base_time = strategy_data["avg_epoch_time"].iloc[0]
        speedup = base_time / strategy_data["avg_epoch_time"]
        num_gpus = strategy_data["num_gpus"]

        marker = next(marker_cycle)
        ax.plot(num_gpus, speedup, marker=marker, lw=1.0, label=strategy, alpha=0.7)

    # Plotting the linear line
    num_gpus = np.array(avg_epoch_time_df["num_gpus"].unique())
    linear_speedup = np.array(avg_epoch_time_df["linear_speedup"].unique())
    ax.plot(
        num_gpus, linear_speedup, ls="dashed", lw=1.0, c="k", label="linear speedup"
    )

    # The log scale must be set before changing the ticks
    ax.set_yscale("log", base=2)
    ax.set_xscale("log", base=2)

    # Making the numbers on the y-axis scalars instead of exponents
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Remove minor ticks on x-axis and format new ticks
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(num_gpus)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_xticks(num_gpus)
    ax.set_title("Relative Speedup of Average Epoch Time vs Number of GPUs")
    ax.legend(ncol=1)

    num_datapoints = len(num_gpus)
    figure_height, figure_width = calculate_plot_dimensions(
        num_datapoints=num_datapoints
    )
    fig.set_figheight(figure_height)
    fig.set_figwidth(figure_width)

    plt.tight_layout()
    sns.reset_orig()

    return fig, ax


def gpu_bar_plot(
    data_df: pd.DataFrame, plot_title: str, y_label: str, main_column: str
) -> Tuple[Figure, Axes]:
    """Creates a bar plot for the specified data, grouped by strategy and number of GPUs.

    Args:
        data_df (pd.DataFrame): A DataFrame containing the data to plot. Must include the
            columns "strategy", "num_global_gpus", and the column specified in
            `main_column`.
        plot_title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        main_column (str): The column in `data_df` to use for the bar heights.

    Returns:
        Tuple[Figure, Axes]: A tuple containing the matplotlib `Figure` and `Axes` objects
        of the generated bar plot.

    Raises:
        ValueError: If `data_df` is missing required columns.
    """
    sns.set_theme()

    strategies = data_df["strategy"].unique()
    unique_gpu_counts = np.array(data_df["num_global_gpus"].unique())
    color_map = plt.get_cmap("tab10")

    fig, ax = plt.subplots()
    x = np.arange(len(unique_gpu_counts))

    bar_width = 1 / (len(strategies) + 1)
    static_offset = (len(strategies) - 1) / 2
    for strategy_idx, strategy in enumerate(strategies):
        dynamic_bar_offset = strategy_idx - static_offset
        strategy_data = data_df[data_df["strategy"] == strategy]
        color = color_map(strategy_idx % 10)

        # Ensuring the correct spacing of the bars
        strategy_num_gpus = len(strategy_data["num_global_gpus"])

        ax.bar(
            x=x[:strategy_num_gpus] + dynamic_bar_offset * bar_width,
            height=strategy_data[main_column],
            width=bar_width,
            label=strategy,
            color=color,
        )

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_gpu_counts)
    ax.legend(title="Strategy")

    num_datapoints = len(unique_gpu_counts)
    figure_height, figure_width = calculate_plot_dimensions(
        num_datapoints=num_datapoints
    )
    fig.set_figheight(figure_height)
    fig.set_figwidth(figure_width)

    sns.reset_orig()

    return fig, ax


def computation_fraction_bar_plot(
    communication_data_df: pd.DataFrame,
) -> Tuple[Figure, Axes]:
    """Creates a stacked bar plot showing computation and communication fractions for
    different strategies and GPU counts.

    Args:
        communication_data_df (pd.DataFrame): A DataFrame containing the following columns:
            - "strategy": The name of the distributed training strategy.
            - "num_gpus": The number of GPUs used.
            - "computation_fraction": The fraction of time spent on computation.

    Returns:
        Tuple[Figure, Axes]: A tuple containing the matplotlib `Figure` and `Axes` objects
        of the generated plot.

    Raises:
        ValueError: If the DataFrame is missing required columns or has invalid data.
    """
    sns.set_theme()
    color_map = plt.get_cmap("tab10")
    hatch_patterns = ["//", r"\\"]

    strategy_labels = sorted(communication_data_df["strategy"].unique())
    gpu_numbers = sorted(communication_data_df["num_gpus"].unique())
    values = communication_data_df.pivot(
        index="strategy", columns="num_gpus", values="computation_fraction"
    ).to_numpy()

    width = 1 / (len(strategy_labels) + 1)
    complements = 1 - values

    x = np.arange(len(gpu_numbers))
    fig, ax = plt.subplots()

    # Creating an offset to "center" around zero
    static_offset = (len(strategy_labels) - 1) / 2
    for strategy_idx in range(len(strategy_labels)):
        dynamic_bar_offset = strategy_idx - static_offset

        color = color_map(strategy_idx % 10)
        hatch = hatch_patterns[strategy_idx % 2]

        ax.bar(
            x=x + dynamic_bar_offset * width,
            height=values[strategy_idx],
            width=width,
            color=color,
            label=strategy_labels[strategy_idx],
            edgecolor="gray",
            linewidth=0.6,
        )
        ax.bar(
            x=x + dynamic_bar_offset * width,
            height=complements[strategy_idx],
            width=width,
            bottom=values[strategy_idx],
            facecolor="none",
            edgecolor="gray",
            alpha=0.8,
            linewidth=0.6,
            hatch=hatch,
        )

    ax.set_ylabel("Computation fraction")
    ax.set_xlabel("Number of GPUs")
    ax.set_title("Computation vs Communication Time by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_numbers)
    ax.set_ylim(0, 1.1)

    # Adding communication time to the legend
    hatch_patch = Patch(
        facecolor="none", edgecolor="gray", hatch="//", label="Communication"
    )
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [hatch_patch])

    # Dynamically adjusting the width of the figure
    figure_width = max(int(2 * len(gpu_numbers)), 8)
    fig.set_figwidth(figure_width)
    fig.set_figheight(figure_width * 0.8)

    # Resetting so that seaborn's theme doesn't affect other plots
    sns.reset_orig()

    return fig, ax
