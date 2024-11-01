# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credits:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# 
# --------------------------------------------------------------------------------------
# Command-line interface for the itwinai Python library. 
# Example: 
# 
# >>> itwinai --help
# 
# --------------------------------------------------------------------------------------
# 
# NOTE: import libraries in the command's function, not here, as having them here will 
# slow down the CLI commands significantly. 

from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def generate_gpu_data_plots(
    log_dir: str = "scalability-metrics/gpu-energy-data",
    pattern: str = r".*\.csv$",
    plot_dir: str = "plots/",
    do_backup: bool = False,
    backup_dir: str = "backup-scalability-metrics/",
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """Generate GPU energy and utilization plots showing the expenditure for each
    combination of strategy and number of GPUs in Watt hours and total computing
    percentage. Backs up the data used to create the plot if ``backup_dir`` is not None

    Args:
        log_dir: The directory where the csv logs are stored. Defaults to
            ``utilization_logs``.
        pattern: A regex pattern to recognize the file names in the 'log_dir' folder.
            Defaults to ``dataframe_(?:\\w+)_(?:\\d+)\\.csv$``. Set it to 'None' to
            make it None. In this case, it will match all files in the given folder.
        plot_dir: The directory where the resulting plots should be saved. Defaults to
            ``plots/``.
        do_backup: Whether to backup the data used for making the plot or not.
        backup_dir: The path to where the data used to produce the plot should be
            saved.
        experiment_name: The name of the experiment to be used when creating a backup
            of the data used for the plot.
        run_name: The name of the run to be used when creating a backup of the data
            used for the plot.

    """
    import uuid

    from itwinai.scalability import convert_matching_files_to_dataframe
    from itwinai.torch.monitoring.plotting import (
        calculate_average_gpu_utilization,
        calculate_total_energy_expenditure,
        gpu_bar_plot,
    )

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(
            f"The provided log_dir, '{log_dir_path.resolve()}', does not exist."
        )

    plot_dir_path = Path(plot_dir)
    if pattern.lower() == "none":
        pattern = None

    gpu_data_df = convert_matching_files_to_dataframe(
        pattern=pattern, log_dir=log_dir_path
    )

    energy_df = calculate_total_energy_expenditure(gpu_data_df=gpu_data_df)
    utilization_df = calculate_average_gpu_utilization(gpu_data_df=gpu_data_df)

    plot_dir_path.mkdir(parents=True, exist_ok=True)
    energy_plot_path = plot_dir_path / "gpu_energy_plot.png"
    utilization_plot_path = plot_dir_path / "utilization_plot.png"

    energy_fig, _ = gpu_bar_plot(
        data_df=energy_df,
        plot_title="Energy Consumption by Strategy and Number of GPUs",
        y_label="Energy Consumption (Wh)",
        main_column="total_energy_wh",
    )
    utilization_fig, _ = gpu_bar_plot(
        data_df=utilization_df,
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

    if experiment_name is None:
        random_id = str(uuid.uuid4())
        experiment_name = "exp_" + random_id[:6]
    if run_name is None:
        random_id = str(uuid.uuid4())
        run_name = "run_" + random_id[:6]

    backup_path = Path(backup_dir) / experiment_name / run_name / "gpu_data.csv"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_data_df.to_csv(backup_path, index=False)
    print(f"Storing backup file at '{backup_path.resolve()}'.")


@app.command()
def generate_communication_plot(
    log_dir: str = "scalability-metrics/communication-data",
    pattern: str = r"(.+)_(\d+)_(\d+)\.csv$",
    output_file: str = "plots/communication_plot.png",
    do_backup: bool = True,
    backup_dir: str = "backup-scalability-metrics/",
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """Generate stacked plot showing computation vs. communication fraction. Stores it
    to output_file.

    Args:
        log_dir: The directory where the csv logs are stored. Defaults to
            ``profiling_logs``.
        pattern: A regex pattern to recognize the file names in the 'log_dir' folder.
            Defaults to ``profile_(\\w+)_(\\d+)_(\\d+)\\.csv$``. Set it to 'None' to
            make it None. In this case, it will match all files in the given folder.
        output_file: The path to where the resulting plot should be saved. Defaults to
            ``plots/comm_plot.png``.
        do_backup: Whether to backup the data used for making the plot or not.
        backup_dir: The path to where the data used to produce the plot should be
            saved.
        experiment_name: The name of the experiment to be used when creating a backup
            of the data used for the plot.
        run_name: The name of the run to be used when creating a backup of the data
            used for the plot.
    """
    import uuid

    import matplotlib.pyplot as plt

    from itwinai.scalability import convert_matching_files_to_dataframe
    from itwinai.torch.profiling.communication_plot import (
        create_stacked_plot,
        get_comp_fraction_full_array,
    )

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(
            f"The provided directory, '{log_dir_path.resolve()}', does not exist."
        )

    if pattern.lower() == "none":
        pattern = None

    expected_columns = {
        "strategy",
        "num_gpus",
        "global_rank",
        "name",
        "self_cuda_time_total",
    }
    communication_df = convert_matching_files_to_dataframe(
        log_dir=log_dir_path, pattern=pattern, expected_columns=expected_columns
    )
    values = get_comp_fraction_full_array(communication_df, print_table=True)

    strategies = sorted(communication_df["strategy"].unique())
    gpu_numbers = sorted(communication_df["num_gpus"].unique(), key=lambda x: int(x))

    fig, _ = create_stacked_plot(values, strategies, gpu_numbers)

    # TODO: set these dynamically?
    fig.set_figwidth(8)
    fig.set_figheight(6)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    print(f"\nSaved computation vs. communication plot at '{output_path.resolve()}'.")

    if not do_backup:
        return

    if experiment_name is None:
        random_id = str(uuid.uuid4())
        experiment_name = "exp_" + random_id[:6]
    if run_name is None:
        random_id = str(uuid.uuid4())
        run_name = "run_" + random_id[:6]

    backup_path = (
        Path(backup_dir) / experiment_name / run_name / "communication_plot.csv"
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    communication_df.to_csv(backup_path, index=False)
    print(f"Storing backup file at '{backup_path.resolve()}'.")


@app.command()
def generate_scalability_plot(
    pattern: str = "None",
    log_dir: str = "scalability-metrics/epoch-time",
    plot_title: Annotated[
        Optional[str], typer.Option(help=("Plot name."))
    ] = "scalability_plot",
    archive: Annotated[
        Optional[str],
        typer.Option(help=("Archive name to backup the data, without extension.")),
    ] = None,

):
    """
    Generate scalability report merging all CSVs containing epoch time
    records in sub-folders.

    Example:

    >>> itwinai scalability-report --pattern="^epoch.+\\.csv$" --skip-id 0 \\
    >>>     --plot-title "Some title" --archive archive_name

    """
    # TODO: add max depth and path different from CWD

    from itwinai.scalability import (
        archive_data,
        create_absolute_plot,
        create_relative_plot,
        read_scalability_files,
    )

    log_dir_path = Path(log_dir)
    if pattern.lower() == "none":
        pattern = None

    combined_df, csv_files = read_scalability_files(
        pattern=pattern, log_dir=log_dir_path
    )
    print("Merged CSV:")
    print(combined_df)

    avg_time_df = (
        combined_df.drop(columns="epoch_id")
        .groupby(["name", "nodes"])
        .mean()
        .reset_index()
    )
    print("\nAvg over name and nodes:")
    print(avg_time_df.rename(columns=dict(time="avg(time)")))

    plot_png = f"scaling_plot_{plot_title}.png"
    create_absolute_plot(avg_time_df)
    create_relative_plot(avg_time_df)

    if archive is not None:
        archive_data(archive, csv_files, plot_png, avg_time_df)


@app.command()
def sanity_check(
    torch: Annotated[
        Optional[bool], typer.Option(help=("Check also itwinai.torch modules."))
    ] = False,
    tensorflow: Annotated[
        Optional[bool], typer.Option(help=("Check also itwinai.tensorflow modules."))
    ] = False,
    all: Annotated[Optional[bool], typer.Option(help=("Check all modules."))] = False,
):
    """Run sanity checks on the installation of itwinai and its dependencies by trying
    to import itwinai modules. By default, only itwinai core modules (neither torch, nor
    tensorflow) are tested."""
    from itwinai.tests.sanity_check import (
        sanity_check_all,
        sanity_check_slim,
        sanity_check_tensorflow,
        sanity_check_torch,
    )

    all = (torch and tensorflow) or all
    if all:
        sanity_check_all()
    elif torch:
        sanity_check_torch()
    elif tensorflow:
        sanity_check_tensorflow()
    else:
        sanity_check_slim()


@app.command()
def exec_pipeline(
    config: Annotated[
        Path,
        typer.Option(help="Path to the configuration file of the pipeline to execute."),
    ],
    pipe_key: Annotated[
        str,
        typer.Option(
            help=(
                "Key in the configuration file identifying "
                "the pipeline object to execute."
            )
        ),
    ] = "pipeline",
    steps: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Run only some steps of the pipeline. Accepted values are "
                "indices, python slices (e.g., 0:3 or 2:10:100), and "
                "string names of steps."
            )
        ),
    ] = None,
    print_config: Annotated[
        bool, typer.Option(help=("Print config to be executed after overrides."))
    ] = False,
    overrides_list: Annotated[
        Optional[List[str]],
        typer.Option(
            "--override",
            "-o",
            help=(
                "Nested key to dynamically override elements in the "
                "configuration file with the "
                "corresponding new value, joined by '='. It is also possible "
                "to index elements in lists using their list index. "
                "Example: [...] "
                "-o pipeline.init_args.trainer.init_args.lr=0.001 "
                "-o pipeline.my_list.2.batch_size=64 "
            ),
        ),
    ] = None,
):
    """Execute a pipeline from configuration file. Allows dynamic override of fields."""
    # Add working directory to python path so that the interpreter is able
    # to find the local python files imported from the pipeline file
    import os
    import re
    import sys

    from .utils import str_to_slice

    sys.path.append(os.path.dirname(config))
    sys.path.append(os.getcwd())

    # Parse and execute pipeline
    from itwinai.parser import ConfigParser

    overrides_list = overrides_list if overrides_list is not None else []
    overrides = {
        k: v
        for k, v in map(lambda x: (x.split("=")[0], x.split("=")[1]), overrides_list)
    }
    parser = ConfigParser(config=config, override_keys=overrides)
    if print_config:
        import json

        print()
        print("#=" * 15 + " Used configuration " + "#=" * 15)
        print(json.dumps(parser.config, indent=2))
        print("#=" * 50)
        print()
    pipeline = parser.parse_pipeline(pipeline_nested_key=pipe_key)
    if steps:
        if not re.match(r"\d+(:\d+)?(:\d+)?", steps):
            print(f"Looking for step name '{steps}'")
        else:
            steps = str_to_slice(steps)
        pipeline = pipeline[steps]
    pipeline.execute()


@app.command()
def mlflow_ui(
    path: str = typer.Option("ml-logs/", help="Path to logs storage."),
    port: int = typer.Option(5000, help="Port on which the MLFlow UI is listening."),
):
    """Visualize Mlflow logs."""
    import subprocess

    subprocess.run(f"mlflow ui --backend-store-uri {path} --port {port}".split())


@app.command()
def mlflow_server(
    path: str = typer.Option("ml-logs/", help="Path to logs storage."),
    port: int = typer.Option(5000, help="Port on which the server is listening."),
):
    """Spawn Mlflow server."""
    import subprocess

    subprocess.run(f"mlflow server --backend-store-uri {path} --port {port}".split())


@app.command()
def kill_mlflow_server(
    port: int = typer.Option(5000, help="Port on which the server is listening."),
):
    """Kill Mlflow server."""
    import subprocess

    subprocess.run(
        f"kill -9 $(lsof -t -i:{port})".split(), check=True, stderr=subprocess.DEVNULL
    )


if __name__ == "__main__":
    app()
