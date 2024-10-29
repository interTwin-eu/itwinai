import glob
import os
import re
import shutil
import itertools
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Union
from re import compile, Pattern, Match
from itertools import cycle

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

def read_scalability_files(pattern: str, log_dir: Path): 
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

def create_absolute_plot(avg_times): 
    sns.set_theme()
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Use built-in matplotlib color cycle and marker styles
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
    color_marker_combinations = itertools.cycle(zip(colors, markers))
    
    # Plot each unique name with its own color and marker combination
    unique_names = avg_times['name'].unique()
    for name in unique_names:
        color, marker = next(color_marker_combinations)
        data = avg_times[avg_times['name'] == name]
        
        ax.plot(data['nodes'], data['time'], 
                marker=marker, color=color, label=name, linestyle='-', markersize=6)

    # Labeling the axes and setting the title
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Average Time")
    ax.set_title("Average Time vs Number of Nodes")
    
    # Show legend and grid
    ax.legend(title="Method")
    ax.grid(True)
    
    # Save the plot as an image
    output_name ="scaling_plot_avg_times.png" 
    plt.savefig("scaling_plot_avg_times.png")
    print(f"Saving absolute plot to '{output_name}'.")
    sns.reset_orig()


def create_relative_plot(plot_title: str, avg_epoch_time_df: pd.DataFrame):
    sns.set_theme()
    fig, speedup_axis = plt.subplots(1, 1, figsize=(6, 4))
    if plot_title is not None:
        fig.suptitle(plot_title)

    speedup_axis.set_yscale("log")
    speedup_axis.set_xscale("log")

    marker_cycle = cycle("ov^s*dXpD.+12348")

    strategy_names = sorted(set(avg_epoch_time_df.name.values))
    for strategy in strategy_names:
        strategy_data = avg_epoch_time_df[avg_epoch_time_df.name == strategy].drop(columns="name")

        # Derived columns
        strategy_data["num_gpus"] = strategy_data["nodes"] * 4
        strategy_data["ideal_speedup"] = strategy_data["nodes"].astype(float)
        base_time = strategy_data["time"].iloc[0]

        strategy_data["speedup"] = base_time / strategy_data["time"]
        strategy_data["n_workers"] = 1

        # Efficiency calculations
        strategy_data["scaled_sim_time_s"] = strategy_data["time"] * strategy_data["nodes"] * strategy_data["n_workers"]
        base_scaled_time = strategy_data["scaled_sim_time_s"].iloc[0]
        strategy_data["efficiency"] = base_scaled_time / strategy_data["scaled_sim_time_s"]

        speedup_axis.plot(
            strategy_data["num_gpus"].values,
            strategy_data["speedup"].values,
            marker=next(marker_cycle),
            lw=1.0,
            label=strategy,
            alpha=0.7,
        )

    # Plotting the ideal line
    speedup_axis.plot(
        avg_epoch_time_df["num_gpus"].values,
        avg_epoch_time_df["ideal_speedup"].values,
        ls="dashed",
        lw=1.0,
        c="k",
        label="ideal",
    )
    speedup_axis.legend(ncol=1)

    speedup_axis.set_xticks(strategy_data["num_gpus"].values)
    speedup_axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    speedup_axis.set_ylabel("Speedup")
    speedup_axis.set_xlabel("Number of GPUs (4 per node)")
    speedup_axis.grid()

    # Sorted legend
    handles, labels = speedup_axis.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    plt.legend(sorted_handles, sorted_labels)

    # Save path and save
    plot_path = Path(f"scaling_plot_{plot_title}.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", format="png", dpi=300)
    print("Saved scaling plot to:", plot_path)
    sns.reset_orig()

def archive_data(archive, csv_files, plot_path, avg_times): 
    if "/" in archive:
        raise ValueError(
            f"Archive name must NOT contain a path. Received: '{archive}'"
        )
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
