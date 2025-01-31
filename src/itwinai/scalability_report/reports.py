# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from pathlib import Path

from itwinai.scalability_report.data import read_scalability_metrics_from_csv
from itwinai.scalability_report.plot import (
    absolute_avg_epoch_time_plot,
    computation_fraction_bar_plot,
    gpu_bar_plot,
    relative_epoch_time_speedup_plot,
)
from itwinai.scalability_report.utils import (
    calculate_gpu_statistics,
    get_computation_fraction_data,
)


def epoch_time_report(
    epoch_time_dir: Path | str,
    plot_dir: Path | str,
    backup_dir: Path,
    do_backup: bool = False,
) -> None:
    """Generates reports and plots for epoch training times across distributed training
    strategies, including a log-log plot of absolute average epoch times against the
    number of GPUs and a log-log plot of relative speedup as more GPUs are added. The
    function optionally creates backups of the data.

    Args:
        epoch_time_dir (Path | str): Path to the directory containing CSV files with
            epoch time metrics. The files must include the columns "name", "nodes",
            "epoch_id", and "time".
        plot_dir (Path | str): Path to the directory where the generated plots will
            be saved.
        backup_dir (Path): Path to the directory where backups of the data will be stored
            if `do_backup` is True.
        do_backup (bool): Whether to create a backup of the epoch time data in the
            `backup_dir`. Defaults to False.
    """
    if isinstance(epoch_time_dir, str):
        epoch_time_dir = Path(epoch_time_dir)
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)

    epoch_time_expected_columns = {"name", "nodes", "epoch_id", "time"}
    epoch_time_df = read_scalability_metrics_from_csv(
        data_dir=epoch_time_dir, expected_columns=epoch_time_expected_columns
    )

    # Calculate the average time per epoch for each strategy and number of nodes
    avg_epoch_time_df = (
        epoch_time_df.groupby(["name", "nodes"])
        .agg(avg_epoch_time=("time", "mean"))
        .reset_index()
    )

    # Print the resulting table
    formatters = {"avg_epoch_time": "{:.2f} s".format}
    epoch_time_table = avg_epoch_time_df.to_string(index=False, formatters=formatters)
    print(epoch_time_table)

    # Create and save the figures
    absolute_fig, _ = absolute_avg_epoch_time_plot(avg_epoch_time_df=avg_epoch_time_df)
    relative_fig, _ = relative_epoch_time_speedup_plot(
        avg_epoch_time_df=avg_epoch_time_df
    )

    absolute_avg_time_plot_path = plot_dir / "absolute_epoch_time.png"
    relative_speedup_plot_path = plot_dir / "relative_epoch_time_speedup.png"

    absolute_fig.savefig(absolute_avg_time_plot_path)
    relative_fig.savefig(relative_speedup_plot_path)
    print(
        f"Saved absolute average time plot at '{absolute_avg_time_plot_path.resolve()}'."
    )
    print(
        f"Saved relative average time plot at '{relative_speedup_plot_path.resolve()}'."
    )

    if not do_backup:
        return

    backup_dir.mkdir(exist_ok=True, parents=True)
    backup_path = backup_dir / "epoch_time_data.csv"
    epoch_time_df.to_csv(backup_path)
    print(f"Storing backup file at '{backup_path.resolve()}'.")


def gpu_data_report(
    gpu_data_dir: Path | str,
    plot_dir: Path | str,
    backup_dir: Path,
    do_backup: bool = False,
) -> None:
    """Generates reports and plots for GPU energy consumption and utilization across
    distributed training strategies. Includes bar plots for energy consumption and GPU
    utilization by strategy and number of GPUs. The function optionally creates backups
    of the data.

    Args:
        gpu_data_dir (Path | str): Path to the directory containing CSV files with GPU
            data. The files must include the columns "sample_idx", "utilization",
            "power", "local_rank", "node_idx", "num_global_gpus", "strategy", and
            "probing_interval".
        plot_dir (Path | str): Path to the directory where the generated plots will
            be saved.
        backup_dir (Path): Path to the directory where backups of the data will be stored
            if `do_backup` is True.
        do_backup (bool): Whether to create a backup of the GPU data in the `backup_dir`.
            Defaults to False.
    """
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)
    gpu_data_expected_columns = {
        "sample_idx",
        "utilization",
        "power",
        "local_rank",
        "node_idx",
        "num_global_gpus",
        "strategy",
        "probing_interval",
    }
    gpu_data_df = read_scalability_metrics_from_csv(
        data_dir=gpu_data_dir, expected_columns=gpu_data_expected_columns
    )
    gpu_data_statistics_df = calculate_gpu_statistics(
        gpu_data_df=gpu_data_df, expected_columns=gpu_data_expected_columns
    )
    formatters = {
        "total_energy_wh": "{:.2f} Wh".format,
        "utilization": "{:.2f} %".format,
    }
    gpu_data_table = gpu_data_statistics_df.to_string(
        index=False, formatters=formatters
    )
    print(gpu_data_table)

    energy_plot_path = plot_dir / "gpu_energy_plot.png"
    utilization_plot_path = plot_dir / "utilization_plot.png"
    energy_fig, _ = gpu_bar_plot(
        data_df=gpu_data_statistics_df,
        plot_title="Energy Consumption by Strategy and Number of GPUs",
        y_label="Energy Consumption (Wh)",
        main_column="total_energy_wh",
    )
    utilization_fig, _ = gpu_bar_plot(
        data_df=gpu_data_statistics_df,
        plot_title="GPU Utilization by Strategy and Number of GPUs",
        y_label="GPU Utilization (%)",
        main_column="utilization",
    )
    energy_fig.savefig(energy_plot_path)
    utilization_fig.savefig(utilization_plot_path)
    print(f"Saved GPU energy plot at '{energy_plot_path.resolve()}'.")
    print(f"Saved utilization plot at '{utilization_plot_path.resolve()}'.")

    if not do_backup:
        return

    backup_dir.mkdir(exist_ok=True, parents=True)
    backup_path = backup_dir / "gpu_data.csv"
    gpu_data_df.to_csv(backup_path)
    print(f"Storing backup file at '{backup_path.resolve()}'.")


def communication_data_report(
    communication_data_dir: Path | str,
    plot_dir: Path | str,
    backup_dir: Path,
    do_backup: bool = False,
) -> None:
    """Generates reports and plots for communication and computation fractions across
    distributed training strategies. Includes a bar plot showing the fraction of time
    spent on computation vs communication for each strategy and GPU count. The function
    optionally creates backups of the data.

    Args:
        communication_data_dir (Path | str): Path to the directory containing CSV files
            with communication data. The files must include the columns "strategy",
            "num_gpus", "global_rank", "name", and "self_cuda_time_total".
        plot_dir (Path | str): Path to the directory where the generated plot will
            be saved.
        backup_dir (Path): Path to the directory where backups of the data will be stored
            if `do_backup` is True.
        do_backup (bool): Whether to create a backup of the communication data in the
            `backup_dir`. Defaults to False.
    """
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)

    communication_data_expected_columns = {
        "strategy",
        "num_gpus",
        "global_rank",
        "name",
        "self_cuda_time_total",
    }
    communication_data_df = read_scalability_metrics_from_csv(
        data_dir=communication_data_dir,
        expected_columns=communication_data_expected_columns,
    )
    computation_fraction_df = get_computation_fraction_data(communication_data_df)

    formatters = {"computation_fraction": lambda x: "{:.2f} %".format(x * 100)}
    communication_data_table = computation_fraction_df.to_string(
        index=False, formatters=formatters
    )
    print(communication_data_table)

    computation_fraction_plot_path = plot_dir / "computation_fraction_plot.png"
    computation_fraction_fig, _ = computation_fraction_bar_plot(computation_fraction_df)
    computation_fraction_fig.savefig(computation_fraction_plot_path)
    print(
        f"Saved computation fraction plot at '{computation_fraction_plot_path.resolve()}'."
    )

    if not do_backup:
        return

    backup_dir.mkdir(exist_ok=True, parents=True)
    backup_path = backup_dir / "communication_data.csv"
    communication_data_df.to_csv(backup_path)
    print(f"Storing backup file at '{backup_path.resolve()}'.")
