# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import List

from mlflow.tracking import MlflowClient

from itwinai.scalability_report.data import (
    read_epoch_time_from_mlflow,
    read_gpu_metrics_from_mlflow,
    read_profiling_data_from_mlflow,
)
from itwinai.scalability_report.plot import (
    absolute_avg_epoch_time_plot,
    computation_fraction_bar_plot,
    computation_vs_other_bar_plot,
    gpu_bar_plot,
    relative_epoch_time_speedup_plot,
)
from itwinai.scalability_report.utils import (
    calculate_epoch_statistics,
    calculate_gpu_statistics,
    get_computation_fraction_data,
    get_computation_vs_other_data,
)
from itwinai.utils import deprecated

cli_logger = logging.getLogger("cli_logger")


def epoch_time_report(
    plot_dir: Path | str,
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None = None,
    plot_file_suffix: str = ".png",
) -> str | None:
    """Generates reports and plots for epoch training times across distributed training
    strategies, including a log-log plot of absolute average epoch times against the
    number of GPUs and a log-log plot of relative speedup as more GPUs are added.

    Args:
        plot_dir (Path | str): Path to the directory where the generated plots will
            be saved.
        mlflow_client (MlflowClient): MLflow client to interact with the MLflow tracking
            server.
        experiment_name (str): Name of the MLflow experiment to retrieve epoch time data
            from.
        run_names (List[str] | None): List of specific run names to filter the epoch
            time data. If None, all runs in the experiment will be considered.
        plot_file_suffix (str): Suffix for the plot file names. Defaults to ".png".
    Returns:
        str | None: A string representation of the epoch time statistics table, or None if
            no data was found.
    """
    epoch_time_df = read_epoch_time_from_mlflow(
        mlflow_client=mlflow_client,
        experiment_name=experiment_name,
        run_names=run_names,
    )
    if epoch_time_df is None:
        return None

    cli_logger.info("\nAnalyzing Epoch Time Data...")
    epoch_time_expected_columns = {
        "strategy",
        "global_world_size",
        "sample_idx",
        "metric_name",
        "value",
    }
    avg_epoch_time_df = calculate_epoch_statistics(
        epoch_time_df=epoch_time_df,
        expected_columns=epoch_time_expected_columns,
    )

    # Print the resulting table
    formatters = {"avg_epoch_time": "{:.2f} s".format}
    epoch_time_table = avg_epoch_time_df.to_string(index=False, formatters=formatters)

    # Create and save the figures
    absolute_fig, _ = absolute_avg_epoch_time_plot(avg_epoch_time_df=avg_epoch_time_df)
    relative_fig, _ = relative_epoch_time_speedup_plot(avg_epoch_time_df=avg_epoch_time_df)

    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir).resolve()

    absolute_avg_time_plot_path = plot_dir / ("absolute_epoch_time" + plot_file_suffix)
    relative_speedup_plot_path = plot_dir / ("relative_epoch_time_speedup" + plot_file_suffix)

    absolute_fig.savefig(absolute_avg_time_plot_path)
    relative_fig.savefig(relative_speedup_plot_path)
    cli_logger.info(
        f"Saved absolute-average-time plot at '{absolute_avg_time_plot_path.resolve()}'."
    )
    cli_logger.info(
        f"Saved relative-average-time plot at '{relative_speedup_plot_path.resolve()}'."
    )
    return epoch_time_table


def gpu_data_report(
    plot_dir: Path | str,
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None = None,
    plot_file_suffix: str = ".png",
    ray_footnote: str | None = None,
) -> str | None:
    """Generates reports and plots for GPU energy consumption and utilization across
    distributed training strategies. Includes bar plots for energy consumption and GPU
    utilization by strategy and number of GPUs.

    Args:
        plot_dir (Path | str): Path to the directory where the generated plots will
            be saved.
        mlflow_client (MlflowClient): MLflow client to interact with the MLflow tracking
        experiment_name (str): Name of the MLflow experiment to retrieve GPU data from.
        run_names (List[str] | None): List of specific run names to filter the GPU data.
            If None, all runs in the experiment will be considered.
        plot_file_suffix (str): Suffix for the plot file names. Defaults to ".png".
        ray_footnote (str | None): Optional footnote for energy plots containing ray
            strategies. Defaults to None.

    Returns:
        str | None: A string representation of the GPU data statistics table, or None if
            no data is available.
    """
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)

    gpu_data_expected_columns = {
        "metric_name",
        "sample_idx",
        "global_world_size",
        "strategy",
        "probing_interval",
    }

    gpu_data_df = read_gpu_metrics_from_mlflow(
        mlflow_client=mlflow_client, experiment_name=experiment_name, run_names=run_names
    )
    if gpu_data_df is None:
        return None

    cli_logger.info("\nAnalyzing GPU Data...")
    gpu_data_statistics_df = calculate_gpu_statistics(
        gpu_data_df=gpu_data_df, expected_columns=gpu_data_expected_columns
    )
    formatters = {
        "total_energy_wh": "{:.2f} Wh".format,
        "utilization": "{:.2f} %".format,
    }
    gpu_data_table = gpu_data_statistics_df.to_string(index=False, formatters=formatters)

    energy_plot_path = plot_dir / ("gpu_energy_plot" + plot_file_suffix)
    utilization_plot_path = plot_dir / ("utilization_plot" + plot_file_suffix)

    energy_fig, _ = gpu_bar_plot(
        data_df=gpu_data_statistics_df,
        plot_title="Energy Consumption by Framework and Number of GPUs",
        y_label="Energy Consumption (Wh)",
        main_column="total_energy_wh",
        ray_footnote=ray_footnote,
    )
    utilization_fig, _ = gpu_bar_plot(
        data_df=gpu_data_statistics_df,
        plot_title="GPU Utilization by Framework and Number of GPUs",
        y_label="GPU Utilization (%)",
        main_column="utilization",
    )
    energy_fig.savefig(energy_plot_path)
    utilization_fig.savefig(utilization_plot_path)
    cli_logger.info(f"Saved GPU energy plot at '{energy_plot_path.resolve()}'.")
    cli_logger.info(f"Saved utilization plot at '{utilization_plot_path.resolve()}'.")

    return gpu_data_table


@deprecated("Please use `computation_data_report` instead.")
def communication_data_report(
    plot_dir: Path | str,
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None,
    plot_file_suffix: str = ".png",
) -> str | None:
    """Generates reports and plots for communication and computation fractions across
    distributed training strategies. Includes a bar plot showing the fraction of time
    spent on computation vs communication for each strategy and GPU count.

    Args:
        plot_dir (Path | str): Path to the directory where the generated plot will
            be saved.
        mlflow_client (MlflowClient): MLflow client to interact with the MLflow tracking
            server.
        experiment_name (str): Name of the MLflow experiment to retrieve data from.
        run_names (List[str]): List of specific run names to filter the data.
            If None, all runs in the experiment will be considered.
        plot_file_suffix (str): Suffix for the plot file names. Defaults to ".png".
    """
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir).resolve()

    communication_data_expected_columns = {
        "num_gpus",
        "strategy",
        "global_rank",
        "name",
        "self_cuda_time_total",
    }
    communication_data_df = read_profiling_data_from_mlflow(
        mlflow_client,
        experiment_name,
        run_names,
        expected_columns=communication_data_expected_columns,
    )
    if communication_data_df is None:
        return None

    cli_logger.info("\nAnalyzing Communication Data...")
    computation_fraction_df = get_computation_fraction_data(communication_data_df)

    formatters = {"computation_fraction": lambda x: f"{x * 100:.2f} %"}
    communication_data_table = computation_fraction_df.to_string(
        index=False, formatters=formatters
    )

    computation_fraction_plot_path = plot_dir / (
        "computation_vs_communication_plot" + plot_file_suffix
    )
    computation_fraction_fig, _ = computation_fraction_bar_plot(computation_fraction_df)
    computation_fraction_fig.savefig(computation_fraction_plot_path)
    cli_logger.info(
        f"Saved computation fraction plot at '{computation_fraction_plot_path.resolve()}'."
    )

    return communication_data_table


def computation_data_report(
    plot_dir: Path | str,
    mlflow_client: MlflowClient,
    experiment_name: str,
    run_names: List[str] | None = None,
    plot_file_suffix: str = ".png",
) -> str | None:
    """Generates reports and plots for computation and other fractions across
    distributed training strategies. Includes a bar plot showing the fraction of time
    spent on computation vs other for each strategy and GPU count.

    Args:
        plot_dir (Path | str): Path to the directory where the generated plot will
            be saved.
        mlflow_client (MlflowClient): MLflow client to interact with the MLflow tracking
            server.
        experiment_name (str): Name of the MLflow experiment to retrieve data from.
        run_names (List[str] | None): List of specific run names to filter the data.
            If None, all runs in the experiment will be considered.
        plot_file_suffix (str): Suffix for the plot file names. Defaults to ".png".

    Returns:
        str | None: A string representation of the computation data statistics table,
        or None if no data is available.
    """
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)

    computation_data_expected_columns = {
        "strategy",
        "num_gpus",
        "global_rank",
        "name",
        "self_cuda_time_total",
    }

    computation_data_df = read_profiling_data_from_mlflow(
        mlflow_client,
        experiment_name,
        run_names,
        expected_columns=computation_data_expected_columns,
    )
    if computation_data_df is None:
        return None

    cli_logger.info("\nAnalyzing Computation Data...")
    computation_fraction_df = get_computation_vs_other_data(computation_data_df)

    formatters = {"computation_fraction": lambda x: f"{x * 100:.2f} %"}
    computation_data_table = computation_fraction_df.to_string(
        index=False, formatters=formatters
    )

    computation_fraction_plot_path = plot_dir / (
        "computation_vs_other_plot" + plot_file_suffix
    )
    computation_fraction_fig, _ = computation_vs_other_bar_plot(computation_fraction_df)
    computation_fraction_fig.savefig(computation_fraction_plot_path)
    cli_logger.info(
        f"Saved computation fraction plot at '{computation_fraction_plot_path.resolve()}'."
    )

    return computation_data_table
