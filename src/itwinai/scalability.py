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

def read_scalability_files(pattern: str, log_dir: Path): 
    pattern_re = re.compile(pattern)
    all_matching_files = []
    dataframes = []

    for entry in log_dir.iterdir(): 
        if not pattern_re.search(str(entry)):
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


def create_relative_plot(plot_title: str, avg_times): 
    sns.set_theme()
    fig, sp_up_ax = plt.subplots(1, 1, figsize=(6, 4))
    if plot_title is not None:
        fig.suptitle(plot_title)

    sp_up_ax.set_yscale("log")
    sp_up_ax.set_xscale("log")

    markers = iter("ov^s*dXpD.+12348")

    series_names = sorted(set(avg_times.name.values))
    for name in series_names:
        df = avg_times[avg_times.name == name].drop(columns="name")


        df["NGPUs"] = df["nodes"] * 4
        # speedup
        df["Speedup - ideal"] = df["nodes"].astype(float)
        df["Speedup"] = df["time"].iloc[0] / df["time"]
        df["Nworkers"] = 1

        # efficiency
        df["Threadscaled Sim. Time / s"] = df["time"] * df["nodes"] * df["Nworkers"]
        df["Efficiency"] = (
            df["Threadscaled Sim. Time / s"].iloc[0] / df["Threadscaled Sim. Time / s"]
        )

        sp_up_ax.plot(
            df["NGPUs"].values,
            df["Speedup"].values,
            marker=next(markers),
            lw=1.0,
            label=name,
            alpha=0.7,
        )

    sp_up_ax.plot(
        df["NGPUs"].values,
        df["Speedup - ideal"].values,
        ls="dashed",
        lw=1.0,
        c="k",
        label="ideal",
    )
    sp_up_ax.legend(ncol=1)

    sp_up_ax.set_xticks(df["NGPUs"].values)
    sp_up_ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    sp_up_ax.set_ylabel("Speedup")
    sp_up_ax.set_xlabel("NGPUs (4 per node)")
    sp_up_ax.grid()

    # Sort legend
    handles, labels = sp_up_ax.get_legend_handles_labels()
    order = np.argsort(labels)
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plot_png = f"scaling_plot_{plot_title}.png"
    plt.tight_layout()
    plt.savefig(plot_png, bbox_inches="tight", format="png", dpi=300)
    print("Saved scaling plot to: ", plot_png)
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
