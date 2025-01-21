from pathlib import Path

from itwinai.scalability_report.plot import create_relative_plot, create_absolute_plot
from itwinai.scalability_report.data import read_epoch_time_data

def epoch_time_report(epoch_time_dir: Path | str, plot_dir: str | Path) -> None:
    """TODO: docstring """
    if isinstance(epoch_time_dir, str): 
        epoch_time_dir = Path(epoch_time_dir)
    if isinstance(plot_dir, str):
        plot_dir = Path(plot_dir)

    epoch_time_expected_columns = {"name", "nodes", "epoch_id", "time"}
    epoch_time_df = read_epoch_time_data(
        epoch_time_dir, expected_columns=epoch_time_expected_columns
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


def gpu_data_report(dir: Path | str): 
    pass 
