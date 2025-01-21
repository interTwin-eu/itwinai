from pathlib import Path

from itwinai.scalability_report.plot import create_relative_plot, create_absolute_plot
from itwinai.scalability_report.data import read_scalability_metrics_from_csv
from itwinai.scalability_report.utils import calculate_total_energy_expenditure, calculate_average_gpu_utilization


def epoch_time_report(epoch_time_dir: Path | str, plot_dir: str | Path) -> None:
    """TODO: docstring"""
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
    formatters = {"avg_epoch_time": "{:.2f}s".format}
    epoch_time_table = avg_epoch_time_df.to_string(index=False, formatters=formatters)
    print(epoch_time_table)

    create_absolute_plot(avg_epoch_time_df=avg_epoch_time_df, plot_dir=plot_dir)
    create_relative_plot(avg_epoch_time_df=avg_epoch_time_df, plot_dir=plot_dir)


def gpu_data_report(gpu_data_dir: Path | str):
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
    energy_df = calculate_total_energy_expenditure(gpu_data_df=gpu_data_df)
    utilization_df = calculate_average_gpu_utilization(gpu_data_df=gpu_data_df)
