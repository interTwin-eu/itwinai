from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd
from scipy.constants import hour as SECONDS_IN_HOUR


def check_contains_columns(
    df: pd.DataFrame, expected_columns: Set, file_path: Path | None = None
) -> None:
    """Check is the given DataFrame contains all the expected columns and raises a
    ValueError if not.
    """
    if not expected_columns.issubset(df.columns):
        missing_columns = expected_columns - set(df.columns)
        if file_path is not None:
            raise ValueError(
                f"Invalid data format! DataFrame at '{file_path.resolve}' is missing"
                f" some necessary columns. \nMissing columns: {missing_columns}."
            )
        else:
            raise ValueError(
                f"Invalid data format for given DataFrame. \nMissing columns:"
                f"{missing_columns}."
            )


def check_probing_interval_consistency(gpu_data_df: pd.DataFrame) -> None:
    """Checks that the probing_interval is consistent within each group of strategy
    and number of GPUs.

    Raises:
        ValueError: If the probing intervals are inconsistent for any group.
    """
    unique_intervals = gpu_data_df.groupby(["strategy", "num_global_gpus"])[
        "probing_interval"
    ].nunique()

    if (unique_intervals > 1).any():
        inconsistent_group = unique_intervals.max()
        raise ValueError(
            f"probing_interval must have the same value for each strategy and "
            f"number of GPUs, but was inconsistent for strategy: {inconsistent_group[0]} "
            f"and number of GPUs: {inconsistent_group[1]}."
        )


def calculate_gpu_statistics(
    gpu_data_df: pd.DataFrame, expected_columns: Set
) -> pd.DataFrame:
    """Calculates both the total energy expenditure (in Watt-hours) and the average GPU
    utilization for each strategy and number of GPUs. Ensures consistent probing intervals.

    Returns:
        pd.DataFrame: A DataFrame containing the total energy expenditure and
            average GPU utilization for each strategy and number of GPUs, with
            the columns ``strategy``, ``num_global_gpus``, ``total_energy_wh``,
            and ``utilization``.

    Raises:
        ValueError: If the given DataFrame does not contain the expected columns.
        ValueError: If the probing intervals are inconsistent for any group.
    """
    check_contains_columns(df=gpu_data_df, expected_columns=expected_columns)
    check_probing_interval_consistency(gpu_data_df)

    # Calculate total energy expenditure and average utilization
    gpu_data_df["energy_wh"] = (
        gpu_data_df["power"] * gpu_data_df["probing_interval"] / SECONDS_IN_HOUR
    )
    aggregated_df = (
        gpu_data_df.groupby(["strategy", "num_global_gpus"])
        .agg(
            total_energy_wh=("energy_wh", "sum"),
            utilization=("utilization", "mean"),
        )
        .reset_index()
    )

    return aggregated_df


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
