import matplotlib
from torch._dynamo.skipfiles import comptime

# Doing this because otherwise I get an error about X11 Forwarding which I believe
# is due to the server trying to pass the image to the client computer
matplotlib.use("Agg")

import argparse
import re
from pathlib import Path
from re import Pattern

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
    re_pattern: Pattern = re.compile(pattern)
    dataframes = []
    for entry in logs_dir.iterdir():
        match = re_pattern.search(str(entry))
        if not match:
            continue

        # Getting the captured regex groups, e.g. the contents of "(\d+)"
        strategy, num_gpus, global_rank = match.groups()
        df = pd.read_csv(entry)
        df["num_gpus"] = num_gpus
        df["global_rank"] = global_rank
        df["strategy"] = strategy

        dataframes.append(df)

    df = pd.concat(dataframes)
    return df


def get_comp_fraction_full_array(df: pd.DataFrame, print_table: bool = False) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--filename",
        type=str,
        default="plots/comm_plot.png",
        help="Location to store the plot",
    )
    args = parser.parse_args()

    logs_dir = Path("profiling_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    pattern = r"profile_(\w+)_(\d+)_(\d+)\.csv$"
    df = create_combined_comm_overhead_df(logs_dir=logs_dir, pattern=pattern)
    values = get_comp_fraction_full_array(df, print_table=True)

    strategies = df["strategy"].unique()
    gpu_numbers = sorted(df["num_gpus"].unique(), key=lambda x: int(x))

    # Generating and showing the plot
    fig, _ = create_stacked_plot(values, strategies, gpu_numbers)
    fig.set_figwidth(8)
    fig.set_figheight(6)

    output_path = Path(args.filename)
    plt.savefig(output_path)
    print(f"\nSaved computation vs. communication plot at '{output_path}'")


if __name__ == "__main__":
    main()
