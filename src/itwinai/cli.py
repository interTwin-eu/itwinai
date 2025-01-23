# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
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
def generate_scalability_report(
    log_dir: str = "scalability-metrics",
    plot_dir: str = "plots",
    do_backup: bool = False,
    backup_root_dir: str = "backup-scalability-metrics/",
    experiment_name: str | None = None,
):
    """Generates scalability reports for epoch time, GPU data, and communication data
    based on log files in the specified directory. Optionally, backups of the reports
    can be created.

    This command processes log files stored in specific subdirectories under the given
    `log_dir`. It generates plots and metrics for scalability analysis and saves them
    in the `plot_dir`. If backups are enabled, the generated reports will also be
    copied to a backup directory under `backup_root_dir`.

    Args:
        log_dir (str): Path to the directory containing log files for scalability metrics.
            Defaults to "scalability-metrics".
        plot_dir (str): Path to the directory where plots and reports will be saved.
            Defaults to "plots".
        do_backup (bool): Whether to create a backup of the generated reports.
            Defaults to False.
        backup_root_dir (str): Root directory where backups will be saved if `do_backup`
            is True. Defaults to "backup-scalability-metrics/".
        experiment_name (str | None): Name of the experiment for identifying backups.
            If None, a unique name is generated. Defaults to None.

    Raises:
        ValueError: If the provided `log_dir` does not exist.

    """
    import uuid

    from itwinai.scalability_report.reports import (
        communication_data_report,
        epoch_time_report,
        gpu_data_report,
    )

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(
            f"The provided log_dir, '{log_dir_path.resolve()}', does not exist."
        )
    plot_dir_path = Path(plot_dir)
    plot_dir_path.mkdir(exist_ok=True, parents=True)

    report_dirs = {
        "Epoch Time": {
            "dir": log_dir_path / "epoch-time",
            "func": epoch_time_report,
        },
        "GPU Data": {
            "dir": log_dir_path / "gpu-energy-data",
            "func": gpu_data_report,
        },
        "Communication Data": {
            "dir": log_dir_path / "communication-data",
            "func": communication_data_report,
        },
    }

    # Setting the backup directory from exp name and run name
    experiment_name = experiment_name or f"exp_{uuid.uuid4().hex[:6]}"
    backup_dir = Path(backup_root_dir) / experiment_name

    # Creating reports from dictionary
    for report_name, details in report_dirs.items():
        report_dir = details["dir"]
        report_func = details["func"]

        if report_dir.exists():
            print("#" * 8, f"{report_name} Report", "#" * 8)
            report_func(
                report_dir,
                plot_dir=plot_dir_path,
                backup_dir=backup_dir,
                do_backup=do_backup,
            )
            print()
        else:
            print(
                f"No report was created for {report_name} as '{report_dir.resolve()}' does "
                f"not exist."
            )


@app.command()
def sanity_check(
    torch: Annotated[
        Optional[bool], typer.Option(help=("Check also itwinai.torch modules."))
    ] = False,
    tensorflow: Annotated[
        Optional[bool], typer.Option(help=("Check also itwinai.tensorflow modules."))
    ] = False,
    all: Annotated[Optional[bool], typer.Option(help=("Check all modules."))] = False,
    optional_deps: List[str] = typer.Option(
        None, help="List of optional dependencies."
    ),
):
    """Run sanity checks on the installation of itwinai and its dependencies by trying
    to import itwinai modules. By default, only itwinai core modules (neither torch, nor
    tensorflow) are tested."""
    from itwinai.tests.sanity_check import (
        run_sanity_check,
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

    if optional_deps is not None:
        run_sanity_check(optional_deps)


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
