"""
Command line interface for out Python application.
You can call commands from the command line.
Example

>>> $ itwinai --help

"""

# NOTE: import libs in the command"s function, not here.
# Otherwise this will slow the whole CLI.

from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def generate_gpu_energy_plot(
    log_dir: str = "scalability_metrics/gpu_energy_data",
    pattern: str = r"gpu_energy_data.*\.csv$",
    output_file: str = "plots/gpu_energy_plot.png",
) -> None:
    """Generate a GPU energy plot showing the expenditure for each combination of
    strategy and number of GPUs in Watt hours.

    Args:
        log_dir: The directory where the csv logs are stored. Defaults to
            ``utilization_logs``.
        pattern: A regex pattern to recognize the file names in the 'log_dir' folder.
            Defaults to ``dataframe_(?:\\w+)_(?:\\d+)\\.csv$``. Set it to 'None' to
            make it None. In this case, it will match all files in the given folder.
        output_file: The path to where the resulting plot should be saved. Defaults to
            ``plots/gpu_energy_plot.png``.

    """
    import matplotlib.pyplot as plt
    from itwinai.torch.monitoring.plotting import gpu_energy_plot, read_energy_df

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(
            f"The provided log_dir, '{log_dir_path.resolve()}', does not exist."
        )

    if pattern.lower() == "none":
        pattern = None

    gpu_utilization_df = read_energy_df(pattern=pattern, log_dir=log_dir_path)
    gpu_energy_plot(gpu_utilization_df=gpu_utilization_df)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    print(f"\nSaved GPU energy plot at '{output_path.resolve()}'.")


@app.command()
def generate_communication_plot(
    log_dir: str = "scalability_metrics/communication_data",
    pattern: str = r"(.+)_(\d+)_(\d+)\.csv$",
    output_file: str = "plots/communication_plot.png",
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
    """
    import matplotlib.pyplot as plt

    from itwinai.torch.profiling.communication_plot import (
        create_combined_comm_overhead_df,
        create_stacked_plot,
        get_comp_fraction_full_array,
    )

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(
            f"The directory '{log_dir_path.resolve()}' does not exist, so could not"
            f"extract profiling logs. Make sure you are running this command in the "
            f"same directory as the logging dir or are passing a sufficient relative"
            f"path."
        )

    if pattern.lower() == "none":
        pattern = None

    df = create_combined_comm_overhead_df(log_dir=log_dir_path, pattern=pattern)
    values = get_comp_fraction_full_array(df, print_table=True)

    strategies = sorted(df["strategy"].unique())
    gpu_numbers = sorted(df["num_gpus"].unique(), key=lambda x: int(x))

    fig, _ = create_stacked_plot(values, strategies, gpu_numbers)

    # TODO: set these dynamically?
    fig.set_figwidth(8)
    fig.set_figheight(6)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    print(f"\nSaved computation vs. communication plot at '{output_path.resolve()}'.")


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
def scalability_report(
    pattern: Annotated[
        str, typer.Option(help="Python pattern matching names of CSVs in sub-folders.")
    ],
    log_dir: Annotated[ 
        str, typer.Option(help="Directory location for the data files to read") 
    ],
    plot_title: Annotated[Optional[str], typer.Option(help=("Plot name."))] = None,
    # skip_id: Annotated[Optional[int], typer.Option(help=("Skip epoch ID."))] = None,
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

    from itwinai.scalability import read_scalability_files, archive_data, create_relative_plot, create_absolute_plot
    log_dir_path = Path(log_dir)

    combined_df, csv_files = read_scalability_files(
        pattern=pattern, log_dir=log_dir_path
    ) 
    print("Merged CSV:")
    print(combined_df)

    avg_times = (
        combined_df.drop(columns="epoch_id")
        .groupby(["name", "nodes"])
        .mean()
        .reset_index()
    )
    print("\nAvg over name and nodes:")
    print(avg_times.rename(columns=dict(time="avg(time)")))

    plot_png = f"scaling_plot_{plot_title}.png"
    create_absolute_plot(avg_times)
    create_relative_plot(plot_title, avg_times)

    if archive is not None:
        archive_data(archive, csv_files, plot_png, avg_times)


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
