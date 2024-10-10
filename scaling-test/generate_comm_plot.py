import matplotlib

# Doing this because otherwise I get an error about X11 Forwarding which I believe
# is due to the server trying to pass the image to the client computer
matplotlib.use("Agg")

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot import create_stacked_plot
from utils import calculate_comp_and_comm_time

sns.set_theme()


def create_combined_comm_overhead_df(logs_dir: Path, pattern: str) -> pd.DataFrame:
    """Create a combined DataFrame using all the files in a folder matching a
    given RegEx. Note that the pattern has to have three capturing groups, where
    the first one is the strategy, the second one the number of GPUs and the
    third one the global rank of the worker.
    """
    pattern = re.compile(pattern)
    dataframes = []
    for entry in logs_dir.iterdir():
        match = pattern.search(str(entry))
        if not match:
            continue

        # Getting the captured regex groups, i.e. the contents of "(\d+)"
        strategy, num_gpus, global_rank = match.groups()
        df = pd.read_csv(entry)
        df["num_gpus"] = num_gpus
        df["global_rank"] = global_rank
        df["strategy"] = strategy

        dataframes.append(df)

    df = pd.concat(dataframes)
    return df


def get_comp_fraction_full_array(df: pd.DataFrame) -> np.ndarray:
    unique_num_gpus = sorted(df["num_gpus"].unique(), key=lambda x: int(x))
    unique_strategies = df["strategy"].unique()
    values = []

    for strategy in unique_strategies:
        strategy_values = []
        for num_gpus in unique_num_gpus:
            filtered_df = df[
                (df["strategy"] == strategy) & (df["num_gpus"] == num_gpus)
            ]

            # For now we assume that we test all strategies for all sizes, but this might
            # be useful to change later
            assert len(filtered_df) > 0
            comp_time, comm_time = calculate_comp_and_comm_time(df=filtered_df)
            comp_fraction = comp_time / (comp_time + comm_time)
            strategy_values.append(comp_fraction)

            print(
                f"Strategy: {strategy:>10}, "
                f"Num. GPUs: {num_gpus}, "
                f"Comp. time: {comp_time:>5.2f}s, "
                f"Comm. time: {comm_time:>5.2f}s"
            )
        values.append(strategy_values)

    return np.array(values)


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--filename",
        type=str,
        default="plots/comm_plot.png",
        help="Location to store the plot",
    )
    args = parser.parse_args()

    logs_dir = Path("logs")
    pattern = f"profile_(\w+)_(\d+)_(\d+)\.csv$"
    df = create_combined_comm_overhead_df(logs_dir=logs_dir, pattern=pattern)
    values = get_comp_fraction_full_array(df)

    strategies = df["strategy"].unique()
    gpu_numbers = sorted(df["num_gpus"].unique(), key=lambda x: int(x))

    # Generating and showing the plot
    fig, ax = create_stacked_plot(values, strategies, gpu_numbers)
    fig.set_figwidth(8)
    fig.set_figheight(6)

    output_path = Path(args.filename)
    plt.savefig(output_path)
    print(f"\nSaved computation vs. communication plot at '{output_path}'")


if __name__ == "__main__":
    main()
