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

import os
import sys
from pathlib import Path
from typing import List, Optional

import hydra
import typer
from hydra.utils import instantiate
from omegaconf import OmegaConf, errors
from typing_extensions import Annotated

from itwinai.utils import get_root_cause, make_config_paths_absolute

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def generate_scalability_report(
    log_dir: Annotated[
        str,
        typer.Option(help=("Which directory to search for the scalability metrics in.")),
    ] = "scalability-metrics",
    plot_dir: Annotated[
        str, typer.Option(help=("Which directory to save the resulting plots in."))
    ] = "plots",
    do_backup: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to store a backup of the scalability metrics that were used"
                " to make the report or not."
            )
        ),
    ] = False,
    backup_root_dir: Annotated[
        str, typer.Option(help=("Which directory to store the backup files in."))
    ] = "backup-scalability-metrics/",
    experiment_name: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "What to name the experiment in the backup directory."
                " Will be automatically generated if left as None."
            )
        ),
    ] = None,
):
    """Generates scalability reports for epoch time, GPU data, and communication data
    based on log files in the specified directory. Optionally, backups of the reports
    can be created.

    This command processes log files stored in specific subdirectories under the given
    `log_dir`. It generates plots and metrics for scalability analysis and saves them
    in the `plot_dir`. If backups are enabled, the generated reports will also be
    copied to a backup directory under `backup_root_dir`.
    """
    import uuid

    from itwinai.scalability_report.reports import (
        communication_data_report,
        epoch_time_report,
        gpu_data_report,
    )

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(f"The provided log_dir, '{log_dir_path.resolve()}', does not exist.")
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
    optional_deps: List[str] = typer.Option(None, help="List of optional dependencies."),
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



def remove_from_argv(n_args: int):
    assert n_args >= 1
    assert n_args < len(sys.argv)
    sys.argv = [sys.argv[0]] + sys.argv[n_args + 1 :]


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def exec_pipeline():
    """Execute a pipeline from configuration file. Allows dynamic override of fields."""

    # Remove 'exec_pipeline' command from CLI args
    sys.argv = sys.argv[1:]
    # Overwrite hydra default logging behavior
    if "--cfg" not in sys.argv and "-c" not in sys.argv:
        sys.argv.append("hydra.output_subdir=null")

    # Add current working directory to the module search path
    # so hydra will find the objects defined in the config (usually paths relative to config)
    sys.path.append(os.getcwd())

    # Process CLI arguments to handle paths
    sys.argv = make_config_paths_absolute(sys.argv)


    exec_pipeline_with_compose()


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def exec_pipeline_with_compose(cfg):
    """Hydra entry function. Parses a configuration file containing a pipeline definition, and
    instantiates and executes the resulting pipeline object.
    Filters steps if `pipe_steps` is provided, otherwise executes the entire pipeline."""

    pipe_steps = OmegaConf.select(cfg, "pipe_steps", default=None)
    pipe_key = OmegaConf.select(cfg, "pipe_key", default="training_pipeline")

    try:
        cfg = OmegaConf.select(cfg, pipe_key, throw_on_missing=True)
    except errors.MissingMandatoryValue as e:
        e.add_note(
            f"Could not find pipeline key {pipe_key}. Make sure that you provide the full "
            "dotpath to your pipeline key."
        )
        raise e

    if pipe_steps:
        try:
            cfg.steps = [cfg.steps[step] for step in pipe_steps]
            print(f"Successfully selected steps {pipe_steps}")
        except errors.ConfigKeyError as e:
            e.add_note(
                "Could not find all selected steps. Please ensure that all steps exist "
                "and that you provided to the dotpath to them. "
                f"Steps provided: {pipe_steps}."
            )
            raise e
    else:
        print("No steps selected. Executing the whole pipeline.")

    # Instantiate and execute the pipeline
    try:
        pipeline = instantiate(cfg, _convert_="all")
        pipeline.execute()
    except Exception as e:
        root = get_root_cause(e)
        raise e
        raise root


@app.command()
def mlflow_ui(
    path: str = typer.Option("ml-logs/", help="Path to logs storage."),
    port: int = typer.Option(5000, help="Port on which the MLFlow UI is listening."),
    host: str = typer.Option(
        "127.0.0.1",
        help="Which host to use. Switch to '0.0.0.0' to e.g. allow for port-forwarding.",
    ),
):
    """Visualize Mlflow logs."""
    import subprocess

    subprocess.run(f"mlflow ui --backend-store-uri {path} --port {port} --host {host}".split())


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
