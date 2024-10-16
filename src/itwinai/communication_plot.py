from pathlib import Path
from re import Pattern, compile
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
    """Calculates the time spent computing and time spent communicating and
    returns a tuple of these numbers in seconds

    Notes:
      - Assumes that you are running with an NCCL backend.
    """
    if "name" not in df.columns:
        raise ValueError("DataFrame should contain column 'name', but does not!")
    if "self_cuda_time_total" not in df.columns:
        raise ValueError(
            "DataFrame should contain column 'self_cuda_time_total', but does not!"
        )

    nccl_comm_pattern = (
        r"ncclKernel_(?:AllReduce|Broadcast|Reduce|AllGather|ReduceScatter)"
    )
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


def create_stacked_plot(
    values: np.ndarray, strategy_labels: List, gpu_numbers: List
) -> Tuple[Any, Any]:
    """Creates a stacked plot showing values from 0 to 1, where the given value
    will be placed on the bottom and the complement will be placed on top for
    each value in 'values'. Returns the figure and the axis so that the caller can
    do what they want with it, e.g. save to file, change it or just show it.

    Notes:
        - Assumes that the rows of 'values' correspond to the labels in
            'strategy_labels' sorted alphabetically and that the columns
            correspond to the GPU numbers in 'gpu_numbers' sorted numerically
            in ascending order.
    """
    sns.set_theme()

    strategy_labels = sorted(strategy_labels)
    gpu_numbers = sorted(gpu_numbers, key=lambda x: int(x))

    width = 1 / (len(strategy_labels) + 1)
    comp_color = "lightblue"
    comm_color = "lightgreen"
    complements = 1 - values

    x = np.arange(len(gpu_numbers))
    # Create the plot
    fig, ax = plt.subplots()

    # Creating an offset to "center" around zero
    static_offset = len(strategy_labels) / 2 - 0.5
    for strategy_idx in range(len(strategy_labels)):
        dynamic_bar_offset = strategy_idx - static_offset

        # Drawing the stacked bars
        ax.bar(
            x=x + dynamic_bar_offset * width,
            height=values[strategy_idx],
            width=width,
            color=comp_color,
        )
        ax.bar(
            x=x + dynamic_bar_offset * width,
            height=complements[strategy_idx],
            width=width,
            bottom=values[strategy_idx],
            color=comm_color,
        )

        for gpu_idx in range(len(gpu_numbers)):
            # Positioning the labels under the stacks
            if np.isnan(values[strategy_idx, gpu_idx]):
                continue
            dynamic_label_offset = strategy_idx - static_offset
            ax.text(
                x=x[gpu_idx] + dynamic_label_offset * width,
                y=-0.1,
                s=strategy_labels[strategy_idx],
                ha="center",
                va="top",
                fontsize=10,
                rotation=60,
            )

    ax.set_ylabel("Computation fraction")
    ax.set_title("Computation vs Communication Time by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_numbers)
    ax.set_ylim(0, 1)

    # Setting the appropriate colors since the legend is manual
    legend_elements = [
        Patch(facecolor=comm_color, label="Communication"),
        Patch(facecolor=comp_color, label="Computation"),
    ]

    # Positioning the legend outside of the plot to not obstruct
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.80, 1.22),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(top=0.85)
    return fig, ax


def create_combined_comm_overhead_df(logs_dir: Path, pattern: str) -> pd.DataFrame:
    """Reads and combines all files in a folder that matches the given regex pattern
    into a single DataFrame. The files must be formatted as csv files.

    Raises:
        ValueError: If not all expected columns are found in the stored DataFrame.
    """
    re_pattern: Pattern = compile(pattern)
    dataframes = []
    expected_columns = {
        "strategy",
        "num_gpus",
        "global_rank",
        "name",
        "self_cuda_time_total",
    }
    for entry in logs_dir.iterdir():
        match = re_pattern.search(str(entry))
        if not match:
            continue

        df = pd.read_csv(entry)
        if not expected_columns.issubset(df.columns):
            missing_columns = expected_columns - set(df.columns)
            raise ValueError(
                f"Invalid data format! File at '{match.string}' doesn't contain all"
                f" necessary columns. \nMissing columns: {missing_columns}"
            )

        dataframes.append(df)
    if len(dataframes) == 0:
        raise ValueError(
            f"No matching files found in '{logs_dir.resolve()}' for pattern '{pattern}'"
        )
    return pd.concat(dataframes)


def get_comp_fraction_full_array(
    df: pd.DataFrame, print_table: bool = False
) -> np.ndarray:
    """Creates a MxN NumPy array where M is the number of strategies
    and N is the number of GPU configurations. The strategies are sorted
    alphabetically and the GPU configurations are sorted in ascending number
    of GPUs.
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

            # Allows asymmetric testing, i.e. not testing all num gpus and all
            # strategies together
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
