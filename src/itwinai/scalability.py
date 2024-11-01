# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
# 
# Created by: Jarl Sondre Sæther 
# 
# Credits: 
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN Openlab 
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN Openlab
# --------------------------------------------------------------------------------------

import glob
import os
import shutil
from itertools import cycle
from pathlib import Path
from re import Match, Pattern, compile
from typing import List, Optional, Tuple, Union

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
                f"No files matched pattern, '{pattern}', in log_dir, "
                f"{log_dir.resolve()}!"
            )
        raise ValueError(error_message)

    return pd.concat(dataframes)


def read_scalability_files(
    pattern: Optional[str], log_dir: Path
) -> Tuple[pd.DataFrame, List]:
    """Iterates over the given ``log_dir`` and collects all files that match the given
    ``pattern``. If the given pattern is None, it will collect all files in the given
    directory.

    Returns:
        pd.DataFrame: A dataframe containing all the data from the .csv files.
        List: A list containing all the paths of all the files that were used to
            create the combined dataframe. Mainly used to be able to archive files
            in the future.

    """
    all_matching_files = []
    dataframes = []
    re_pattern: Optional[Pattern] = None
    if pattern is not None:
        re_pattern = compile(pattern)

    for entry in log_dir.iterdir():
        match: Union[bool, Match] = True
        if re_pattern is not None:
            match = re_pattern.search(str(entry))

        if not match:
            continue

        all_matching_files.append(entry.resolve())
        df = pd.read_csv(entry)
        dataframes.append(df)

    combined_df = pd.concat(dataframes)
    return combined_df, all_matching_files


def create_absolute_plot(avg_epoch_time_df: pd.DataFrame) -> None:
    """Creates a plot showing the absolute training times for the different
    distributed strategies and different number of GPUs.
    """
    sns.set_theme()
    fig, ax = plt.subplots()

    marker_cycle = cycle("ov^s*dXpD.+12348")

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

    # Labeling the axes and setting the title
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Average Time")
    ax.set_title("Average Time vs Number of Nodes")

    # Show legend and grid
    ax.legend(title="Method")
    ax.grid(True)

    # Save the plot as an image
    output_path = Path("plots/absolute_scalability_plot.png")
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


def archive_data(archive, csv_files, plot_path, avg_times):
    if "/" in archive:
        raise ValueError(f"Archive name must NOT contain a path. Received: '{archive}'")
    if "." in archive:
        raise ValueError(
            f"Archive name must NOT contain an extension. Received: '{archive}'"
        )
    if os.path.isdir(archive):
        raise ValueError(f"Folder '{archive}' already exists. Change archive name.")
    os.makedirs(archive)
    for csvfile in csv_files:
        shutil.copyfile(csvfile, os.path.join(archive, os.path.basename(csvfile)))
    shutil.copyfile(plot_path, os.path.join(archive, plot_path))
    avg_times.to_csv(os.path.join(archive, "avg_times.csv"), index=False)
    print("Archived AVG epoch times CSV")

    # Copy SLURM logs: *.err *.out files
    if os.path.exists("logs_slurm"):
        print("Archived SLURM logs")
        shutil.copytree("logs_slurm", os.path.join(archive, "logs_slurm"))
    # Copy other SLURM logs
    for ext in ["*.out", "*.err"]:
        for file in glob.glob(ext):
            shutil.copyfile(file, os.path.join(archive, file))

    # Create archive
    archive_name = shutil.make_archive(
        base_name=archive,  # archive file name
        format="gztar",
        # root_dir='.',
        base_dir=archive,  # folder path inside archive
    )
    shutil.rmtree(archive)
    print("Archived logs and plot at: ", archive_name)
