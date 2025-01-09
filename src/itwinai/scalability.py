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
from itertools import cycle
from pathlib import Path
from re import Match, Pattern, compile
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def convert_matching_files_to_dataframe(
    log_dir: Path, pattern: Optional[str], expected_columns: Optional[set] = None
) -> pd.DataFrame:
    """Reads and combines all files in a folder that matches the given regex pattern
    into a single DataFrame. The files must be formatted as csv files. If pattern is
    None, we assume a match on all files.

    Raises:
        ValueError: If not all expected columns are found in the stored DataFrame.
        ValueError: If no matching files are found in the given logging directory.
    """
    re_pattern: Optional[Pattern] = None
    if pattern is not None:
        re_pattern = compile(pattern)

    if expected_columns is None:
        expected_columns = set()

    dataframes = []
    for entry in log_dir.iterdir():
        match: Union[bool, Match] = True
        if re_pattern is not None:
            match = re_pattern.search(str(entry))

        if not match:
            continue

        df = pd.read_csv(entry)
        if not expected_columns.issubset(df.columns):
            missing_columns = expected_columns - set(df.columns)
            raise ValueError(
                f"Invalid data format! File at '{str(entry)}' doesn't contain all"
                f" necessary columns. \nMissing columns: {missing_columns}"
            )

        dataframes.append(df)

    if len(dataframes) == 0:
        if pattern is None:
            error_message = f"Unable to find any files in {log_dir.resolve()}!"
        else:
            error_message = (
                f"No files matched pattern, '{pattern}', in log_dir, " f"{log_dir.resolve()}!"
            )
        raise ValueError(error_message)

    return pd.concat(dataframes)


def create_absolute_plot(avg_epoch_time_df: pd.DataFrame) -> None:
    """Creates a plot showing the absolute training times for the different
    distributed strategies and different number of GPUs.
    """
    sns.set_theme()
    fig, ax = plt.subplots()

    marker_cycle = cycle("ov^s*dXpD.+12348")

    unique_nodes = list(avg_epoch_time_df["nodes"].unique())
    unique_names = avg_epoch_time_df["name"].unique()
    for name in unique_names:
        # color, marker = next(color_marker_combinations)
        marker = next(marker_cycle)
        data = avg_epoch_time_df[avg_epoch_time_df["name"] == name]

        ax.plot(
            data["nodes"],
            data["time"],
            marker=marker,
            label=name,
            linestyle="-",
            markersize=6,
        )

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_xticks(unique_nodes)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Average Time (s)")
    ax.set_title("Average Time vs Number of Nodes")

    ax.legend(title="Method")
    ax.grid(True)

    output_path = Path("plots/absolute_scalability_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saving absolute plot to '{output_path.resolve()}'.")
    sns.reset_orig()


def create_relative_plot(avg_epoch_time_df: pd.DataFrame, gpus_per_node: int = 4):
    """Creates a plot showing the relative training times for the different
    distributed strategies and different number of GPUs. In particular, it shows the
    speedup when adding more GPUs, compared to the baseline of using a single node.
    """
    sns.set_theme()

    fig, ax = plt.subplots(figsize=(6, 4))
    # fig.suptitle(plot_title)

    ax.set_yscale("log")
    ax.set_xscale("log")

    marker_cycle = cycle("ov^s*dXpD.+12348")
    avg_epoch_time_df["num_gpus"] = avg_epoch_time_df["nodes"] * gpus_per_node
    avg_epoch_time_df["linear_speedup"] = avg_epoch_time_df["nodes"].astype(float)

    # Plotting the speedup for each strategy
    strategy_names = sorted(avg_epoch_time_df["name"].unique())
    for strategy in strategy_names:
        strategy_data = avg_epoch_time_df[avg_epoch_time_df.name == strategy]

        base_time = strategy_data["time"].iloc[0]
        speedup = base_time / strategy_data["time"]
        num_gpus = strategy_data["num_gpus"]

        marker = next(marker_cycle)
        ax.plot(num_gpus, speedup, marker=marker, lw=1.0, label=strategy, alpha=0.7)

    # Plotting the linear line
    num_gpus = np.array(avg_epoch_time_df["num_gpus"].unique())
    linear_speedup = np.array(avg_epoch_time_df["linear_speedup"].unique())
    ax.plot(num_gpus, linear_speedup, ls="dashed", lw=1.0, c="k", label="linear speedup")

    ax.legend(ncol=1)
    ax.set_xticks(num_gpus)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("Speedup")
    ax.set_xlabel("Number of GPUs (4 per node)")
    ax.grid(True)

    # Sorted legend
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    plt.legend(sorted_handles, sorted_labels)

    plot_path = Path("plots/relative_scalability_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", format="png", dpi=300)
    print(f"Saving relative plot to '{plot_path.resolve()}'.")

    sns.reset_orig()


def backup_scalability_metrics(
    metric_df: pd.DataFrame,
    experiment_name: Optional[str],
    run_name: Optional[str],
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
