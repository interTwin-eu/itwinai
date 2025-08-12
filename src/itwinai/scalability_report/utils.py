# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from pathlib import Path
from typing import Set, Tuple

import pandas as pd
from scipy.constants import hour as SECONDS_IN_HOUR

from itwinai.utils import deprecated


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


def calculate_gpu_statistics(gpu_data_df: pd.DataFrame, expected_columns: Set) -> pd.DataFrame:
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

    mask = gpu_data_df["metric_name"] == "gpu_power_W"
    # Ensure value and probing_interval are numeric
    gpu_data_df.loc[mask, "value"] = pd.to_numeric(gpu_data_df.loc[mask, "value"])
    gpu_data_df.loc[mask, "probing_interval"] = pd.to_numeric(
        gpu_data_df.loc[mask, "probing_interval"]
    )

    # Calculate energy in watt hours
    gpu_data_df.loc[mask, "energy_wh"] = (
        gpu_data_df.loc[mask, "value"]
        * gpu_data_df.loc[mask, "probing_interval"]
        / SECONDS_IN_HOUR
    )

    # shift metrics to columns (assumes samples are the same for each strategy and
    # num_global_gpus), ensured earlier by check_probing_interval_consistency
    pivoted = gpu_data_df.pivot_table(
        index=["strategy", "num_global_gpus", "timestamp"],
        columns="metric_name",
        values="value",
    ).reset_index()

    # Merge previous columns into the pivoted DataFrame
    pivoted = pivoted.merge(
        gpu_data_df[["strategy", "num_global_gpus", "timestamp", "energy_wh"]],
        how="left",
        on=["strategy", "num_global_gpus", "timestamp"],
    )

    # Aggregate as before
    aggregated_df = (
        pivoted.groupby(["strategy", "num_global_gpus"])
        .agg(
            total_energy_wh=("energy_wh", "sum"),  # Total energy in watt-hours
            utilization=("gpu_utilization_percent", "mean"),  # Average GPU utilization
        )
        .reset_index()
    )

    return aggregated_df


@deprecated(
    "Communication vs computation is unreliable and comparable between GPU"
    " architectures. Please use calculate_comp_time instead."
)
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
        "all_reduce",
        "broadcast",
        "reduce",
        "all_gather",
        "gather",
        "reduce_scatter",
    ]
    nccl_comm_pattern = rf"nccl:(?:{'|'.join(comm_types)})"
    cuda_stream_pattern = r"cudaStream(?:WaitEvent|Synchronize)"

    # Any operation that is a part of PyTorch's ATen library and autograd is considered a
    # computation.
    # See torch namespaces: https://docs.pytorch.org/cppdocs/api/library_root.html
    aten_comp_pattern = r".*(?:aten|\sat|c10|autograd)::"

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


def calculate_comp_time(df: pd.DataFrame) -> float:
    """Calculates the time spent on computation in seconds from the
    given DataFrame.

    Raises:
        ValueError: If the DataFrame is missing the required columns 'name' or
        'self_cuda_time_total'.
    """
    expected_columns = {"name", "self_cuda_time_total"}
    check_contains_columns(df=df, expected_columns=expected_columns)

    # Any operation that is a part of PyTorch's ATen library and autograd is considered a
    # computation.
    # See torch namespaces: https://docs.pytorch.org/cppdocs/api/library_root.html
    aten_comp_pattern = r".*(?:aten|\sat|c10|autograd)::"

    comp_df = df[df["name"].str.contains(aten_comp_pattern)]

    comp_time = comp_df["self_cuda_time_total"].sum()

    # Converting from microseconds to seconds
    comp_time *= 1e-6

    return comp_time


@deprecated(
    "Communication calculation is unreliable and not comparable between GPU"
    " architectures. Please use get_computation_vs_other_data instead."
)
def get_computation_fraction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the computation fraction for each strategy and GPU configuration,
    returning a DataFrame with the results. The computation fraction is defined as the
    ratio of computation time to the total time (computation + communication).
    """

    # Group by strategy and number of GPUs, calculate computation fraction
    def compute_fraction(group):
        comp_time, comm_time = calculate_comp_and_comm_time(df=group)
        return comp_time / (comp_time + comm_time + 1e-10)

    grouped = df.groupby(["strategy", "num_gpus"]).apply(compute_fraction)

    # Sort and create cartesian product of unique strategies and GPU counts
    unique_num_gpus = sorted(df["num_gpus"].unique(), key=lambda x: int(x))
    unique_strategies = sorted(df["strategy"].unique())
    index = pd.MultiIndex.from_product(
        [unique_strategies, unique_num_gpus], names=["strategy", "num_gpus"]
    )
    # Reindex to fill in missing combinations with NaN
    result_df = pd.DataFrame(grouped.reindex(index))
    result_df = result_df.reset_index()
    result_df.columns = ["strategy", "num_gpus", "computation_fraction"]
    return result_df


def get_computation_vs_other_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the computation fraction for each strategy and GPU configuration,
    returning a DataFrame with the results. The computation fraction is defined as the
    ratio of computation time to the total time of profiling.
    """

    # Group by strategy and number of GPUs, calculate computation fraction
    def compute_fraction(group):
        # Theoretically, two identical float32 runtimes could be the same and thus disregarded
        # but this is impossible to happen in practice for ml runs.

        # Convert from microseconds to seconds
        total_time = group["self_cuda_time_total"].sum() * 1e-6
        profiler_overhead = group.loc[
            group["name"] == "ProfilerStep*", "self_cuda_time_total"
        ].values
        profiler_overhead = profiler_overhead.sum() if len(profiler_overhead) > 0 else 0.0
        profiler_overhead *= 1e-6  # Convert from microseconds to seconds

        total_time_without_profiler = total_time - profiler_overhead
        comp_time = calculate_comp_time(df=group)

        return comp_time / (total_time_without_profiler + 1e-10)

    grouped = df.groupby(["num_gpus", "strategy"]).apply(compute_fraction)

    # Sort and create cartesian product of unique strategies and GPU counts
    unique_num_gpus = sorted(df["num_gpus"].unique(), key=lambda x: int(x))
    unique_strategies = sorted(df["strategy"].unique())
    index = pd.MultiIndex.from_product(
        [unique_num_gpus, unique_strategies], names=["num_gpus", "strategy"]
    )
    # Reindex to fill in missing combinations with NaN
    result_df = pd.DataFrame(grouped.reindex(index))
    result_df = result_df.reset_index()
    result_df.columns = ["num_gpus", "strategy", "computation_fraction"]

    return result_df
