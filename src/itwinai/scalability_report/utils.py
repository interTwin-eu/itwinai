# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from pathlib import Path
from typing import Set, Tuple

import pandas as pd
from scipy.constants import hour as SECONDS_IN_HOUR


def check_contains_columns(
    df: pd.DataFrame, expected_columns: Set, file_path: Path | None = None
) -> None:
    """Validates that the given DataFrame contains all the expected columns. Raises a
    ValueError if any columns are missing, including the file path in the error message
    if provided.
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
    """Calculates the time spent on computation and communication in seconds from the
    given DataFrame, assuming an NCCL backend.

    Raises:
        ValueError: If the DataFrame is missing the required columns 'name' or
        'self_cuda_time_total'.
    """
    expected_columns = {"name", "self_cuda_time_total"}
    check_contains_columns(df=df, expected_columns=expected_columns)
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


def get_computation_fraction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the computation fraction for each strategy and GPU configuration,
    returning a DataFrame with the results. The computation fraction is defined as the
    ratio of computation time to the total time (computation + communication).
    """
    # Sort and create cartesian product of unique strategies and GPU counts
    unique_num_gpus = sorted(df["num_gpus"].unique(), key=lambda x: int(x))
    unique_strategies = sorted(df["strategy"].unique())
    index = pd.MultiIndex.from_product(
        [unique_strategies, unique_num_gpus], names=["strategy", "num_gpus"]
    )

    # Group by strategy and number of GPUs, calculate computation fraction
    def compute_fraction(group):
        comp_time, comm_time = calculate_comp_and_comm_time(df=group)
        return comp_time / (comp_time + comm_time + 1e-10)

    grouped = df.groupby(["strategy", "num_gpus"]).apply(compute_fraction)

    # Reindex to fill in missing combinations with NaN
    result_df = pd.DataFrame(grouped.reindex(index))
    result_df = result_df.reset_index()
    result_df.columns = ["strategy", "num_gpus", "computation_fraction"]
    return result_df
