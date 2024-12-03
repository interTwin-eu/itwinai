# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Doing this because otherwise I get an error about X11 Forwarding which I believe
# is due to the server trying to pass the image to the client computer
matplotlib.use("Agg")


def calculate_comp_and_comm_time(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculates the time spent computing and time spent communicating and returns a
    tuple of these numbers in seconds. Assumes that you are running with an NCCL
    backend.

    Raises:
        ValueError: If not all expected columns ('name', 'self_cuda_time_total') are
            found in the given DataFrame.
    """
    expected_columns = {"name", "self_cuda_time_total"}
    if not expected_columns.issubset(df.columns):
        missing_columns = expected_columns - set(df.columns)
        raise ValueError(
            f"Invalid data format! DataFrame does not contain the necessary columns."
            f"\nMissing columns: {missing_columns}"
        )

    comm_types = [
        "AllReduce",
        "Broadcast",
        "Reduce",
        "AllGather",
        "Gather",
        "ReduceScatter",
    ]
    nccl_comm_pattern = rf"(?:{'|'.join(comm_types)})"
    cuda_stream_pattern = r"cudaStream(?:WaitEvent|Synchronize)"

    # Any operation that is a part of PyTorch's ATen library is considered a computation
    aten_comp_pattern = r"aten::"

    comm_df = df[
        (df["name"].str.contains(nccl_comm_pattern))
        | (df["name"].str.contains(cuda_stream_pattern))
    ]
    comp_df = df[df["name"].str.contains(aten_comp_pattern)]

    comp_time = comp_df["self_cuda_time_total"].sum()
    comm_time = comm_df["self_cuda_time_total"].sum()

    # Converting from microseconds to seconds
    comp_time *= 1e-6
    comm_time *= 1e-6

    return comp_time, comm_time


def communication_overhead_stacked_bar_plot(
    values: np.ndarray, strategy_labels: List, gpu_numbers: List
) -> Tuple[Any, Any]:
    """Creates a stacked plot showing values from 0 to 1, where the given value
    will be placed on the bottom and the complement will be placed on top for
    each value in 'values'. Returns the figure and the axis so that the caller can
    do what they want with it, e.g. save to file, change it or just show it.

    Notes:
        - Assumes that the rows of 'values' correspond to the labels in
            'strategy_labels' sorted alphabetically and that the columns correspond to
            the GPU numbers in 'gpu_numbers' sorted numerically in ascending order.
    """
    sns.set_theme()
    color_map = plt.get_cmap("tab10")
    hatch_patterns = ["//", r"\\"]

    strategy_labels = sorted(strategy_labels)
    gpu_numbers = sorted(gpu_numbers, key=lambda x: int(x))

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


def get_comp_fraction_full_array(
    df: pd.DataFrame, print_table: bool = False
) -> np.ndarray:
    """Creates a MxN NumPy array where M is the number of strategies and N is the
    number of GPU configurations. The strategies are sorted alphabetically and the GPU
    configurations are sorted in ascending number of GPUs.
    """
    unique_num_gpus = sorted(df["num_gpus"].unique(), key=lambda x: int(x))
    unique_strategies = sorted(df["strategy"].unique())
    values = []

    table_string = ""

    for strategy in unique_strategies:
        strategy_values = []
        for num_gpus in unique_num_gpus:
            filtered_df = df[
                (df["strategy"] == strategy) & (df["num_gpus"] == num_gpus)
            ]

            row_string = f"{strategy:>12} | {num_gpus:>10}"

            # Allows some strategies or num GPUs to not be included
            if len(filtered_df) == 0:
                comp_time, comm_time = np.NaN, np.NaN
                strategy_values.append(np.NaN)

                row_string += f" | {'(NO DATA)':>15}"
            else:
                comp_time, comm_time = calculate_comp_and_comm_time(df=filtered_df)
                # Avoid division-by-zero errors (1e-10)
                comp_fraction = comp_time / (comp_time + comm_time + 1e-10)
                strategy_values.append(comp_fraction)

                row_string += f" | {comp_time:>8.2f}s"
                row_string += f" | {comm_time:>8.2f}s"

            table_string += row_string + "\n"
        values.append(strategy_values)

    if print_table:
        print(f"{'-'*50}")
        print(f"{'Strategy':>12} | {'Num. GPUs':>10} | {'Comp.':>9} | {'Comm.':>8}")
        print(f"{'-'*50}")
        print(table_string)
        print(f"{'-'*50}")

    return np.array(values)
