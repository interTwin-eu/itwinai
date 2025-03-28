# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
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
from typing_extensions import Annotated

from itwinai.utils import make_config_paths_absolute

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def generate_py_spy_report(
    file: Annotated[str, typer.Option(help="The location of the raw profiling data.")],
    num_rows: Annotated[
        str,
        typer.Option(help="Number of rows to display. Pass 'all' to print the full table."),
    ] = "10",
):
    from itwinai.torch.profiling.py_spy_aggregation import (
        handle_data_point,
        create_bottom_function_table,
    )

    if not num_rows.isnumeric() and num_rows != "all":
        raise typer.BadParameter(
            f"'num-rows' has to be either an integer or 'all'. Was '{num_rows}'."
        )
    parsed_num_rows: int | None = int(num_rows) if num_rows.isnumeric() else None
    if isinstance(parsed_num_rows, int) and parsed_num_rows < 1:
        raise typer.BadParameter(f"'num_rows' has to be greater than one! Was '{num_rows}'.")

    file_path = Path(file)
    with file_path.open("r") as f:
        profiling_data = f.readlines()

    data_points = [handle_data_point(line) for line in profiling_data]
    data_points = [dp for dp in data_points if dp]
    leaf_functions = [data_point[-1] for data_point in data_points]
    leaf_functions.sort(key=lambda x: x["num_samples"], reverse=True)
    table = create_bottom_function_table(
        function_data_list=leaf_functions, num_rows=parsed_num_rows
    )
    print(table)


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
    plot_file_suffix: Annotated[
        str,
        typer.Option(
            help=(
                "Which file suffix to use for the plots. Useful for changing between raster"
                " and vector based images"
            )
        ),
    ] = ".png",
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
                plot_file_suffix=plot_file_suffix,
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


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def generate_slurm(
    job_name: Annotated[
        str | None, typer.Option("--job-name", help="The name of the SLURM job.")
    ] = None,
    account: Annotated[
        str, typer.Option("--account", help="The billing account for the SLURM job.")
    ] = "intertwin",
    time: Annotated[
        str, typer.Option("--time", help="The time limit of the SLURM job.")
    ] = "00:30:00",
    partition: Annotated[
        str,
        typer.Option(
            "--partition",
            help="Which partition of the cluster the SLURM job is going to run on.",
        ),
    ] = "develbooster",
    std_out: Annotated[
        str | None, typer.Option("--std-out", help="The standard out file.")
    ] = None,
    err_out: Annotated[
        str | None, typer.Option("--err-out", help="The error out file.")
    ] = None,
    num_nodes: Annotated[
        int,
        typer.Option(
            "--num-nodes",
            help="The number of nodes that the SLURM job is going to run on.",
        ),
    ] = 1,
    num_tasks_per_node: Annotated[
        int, typer.Option("--num-tasks-per-node", help="The number of tasks per node.")
    ] = 1,
    gpus_per_node: Annotated[
        int,
        typer.Option("--gpus-per-node", help="The requested number of GPUs per node."),
    ] = 4,
    cpus_per_gpu: Annotated[
        int,
        typer.Option("--cpus-per-gpu", help="The requested number of CPUs per GPU."),
    ] = 4,
    config_path: Annotated[
        str,
        typer.Option(
            "--config-path",
            help="The path to the directory containing the config file to use for training.",
        ),
    ] = ".",
    config_name: Annotated[
        str,
        typer.Option("--config-name", help="The name of the config file to use for training."),
    ] = "config",
    pipe_key: Annotated[
        str,
        typer.Option("--pipe-key", help="Which pipe key to use for running the pipeline."),
    ] = "rnn_training_pipeline",
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Which mode to run, e.g. scaling test, all strategies, or a single run.",
            case_sensitive=False,
        ),
    ] = "single",
    dist_strat: Annotated[
        str,
        typer.Option(
            "--dist-strat",
            help="Which distributed strategy to use.",
            case_sensitive=False,
        ),
    ] = "ddp",
    training_cmd: Annotated[
        str | None,
        typer.Option(
            "--training-cmd", help="The training command to use for the python script."
        ),
    ] = None,
    python_venv: Annotated[
        str,
        typer.Option(
            "--python-venv", help="Which python venv to use for running the command."
        ),
    ] = ".venv",
    scalability_nodes: Annotated[
        str,
        typer.Option(
            "--scalability-nodes",
            help="A comma-separated list of node numbers to use for the scalability test.",
        ),
    ] = "1,2,4,8",
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Whether to include debugging information or not"),
    ] = False,
    no_save_script: Annotated[
        bool,
        typer.Option(
            "--no-save-script", help="Whether to save the script after processing it."
        ),
    ] = False,
    no_submit_job: Annotated[
        bool,
        typer.Option(
            "--no-submit-job",
            help="Whether to submit the job when processing the script.",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", help="The path to the SLURM configuration file."),
    ] = None,
):
    """Generates a default SLURM script using arguments and optionally a configuration
    file.
    """
    from itwinai.slurm.slurm_script_builder import generate_default_slurm_script

    del sys.argv[0]
    generate_default_slurm_script()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def exec_pipeline(
    # NOTE: The arguments below are not actually needed in this function, but they are here
    # to replicate Hydra's help page in Typer, making it easier for the users to use it.
    hydra_help: Annotated[bool, typer.Option(help="Show Hydra's help page")] = False,
    version: Annotated[bool, typer.Option(help="Show Hydra's version and exit")] = False,
    cfg: Annotated[
        str,
        typer.Option("--cfg", "-c", help="Show config instead of running [job|hydra|all]"),
    ] = "",
    resolve: Annotated[
        bool,
        typer.Option(
            help="Used in conjunction with --cfg, resolve "
            "config interpolations before printing."
        ),
    ] = False,
    package: Annotated[
        str,
        typer.Option("--package", "-p", help="Config package to show"),
    ] = "",
    run: Annotated[
        str,
        typer.Option("--run", "-r", help="Run a job"),
    ] = "",
    multirun: Annotated[
        str,
        typer.Option(
            "--multirun",
            "-m",
            help="Run multiple jobs with the configured launcher and sweeper",
        ),
    ] = "",
    shell_completion: Annotated[
        str,
        typer.Option(
            "--shell-completion",
            "-sc",
            help="Install or Uninstall shell completion",
        ),
    ] = "",
    config_path: Annotated[
        str,
        typer.Option(
            "--config-path",
            "-cp",
            help=(
                # NOTE: this docstring changed from Hydra's help page.
                "Overrides the config_path specified in hydra.main(). "
                "The config_path is absolute, or relative to the current workign directory. "
                "Defaults to the current working directory."
            ),
        ),
    ] = "",
    config_name: Annotated[
        str,
        typer.Option(
            "--config-name",
            "-cn",
            help="Overrides the config_name specified in hydra.main()",
        ),
    ] = "config",
    config_dir: Annotated[
        str,
        typer.Option(
            "--config-dir",
            "-cd",
            help="Adds an additional config dir to the config search path",
        ),
    ] = "",
    experimental_rerun: Annotated[
        str,
        typer.Option(
            "--experimental-rerun",
            help="Rerun a job from a previous config pickle",
        ),
    ] = "",
    info: Annotated[
        str,
        typer.Option(
            "--info",
            "-i",
            help=(
                "Print Hydra information "
                "[all|config|defaults|defaults-tree|plugins|searchpath]"
            ),
        ),
    ] = "",
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help=(
                "Any key=value arguments to override config values "
                "(use dots for.nested=overrides), using the Hydra syntax."
            ),
        ),
    ] = None,
):
    """Execute a pipeline from configuration file using Hydra CLI. Allows dynamic override
    of fields which can be appended as a list of overrides (e.g., batch_size=32).
    By default, it will expect a configuration file called "config.yaml" in the
    current working directory. To override the default behavior set --config-name and
    --config-path.
    By default, this command will execute the whole pipeline under "training_pipeline"
    field in the configuration file. To execute a different pipeline you can override this
    by passing "+pipe_key=your_pipeline" in the list of overrides, and to execute only a
    subset of the steps, you can pass "+pipe_steps=[0,1]".
    """

    del sys.argv[0]

    # Add current working directory to the module search path
    # so hydra will find the objects defined in the config (usually paths relative to config)
    sys.path.append(os.getcwd())

    # Process CLI arguments to handle paths
    sys.argv = make_config_paths_absolute(sys.argv)

    exec_pipeline_with_compose()


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def exec_pipeline_with_compose(cfg):
    """Hydra entry function. The hydra.main decorator parses a configuration file
    (under config_path), which contains a pipeline definition, and passes it to this function
    as an omegaconf.DictConfig object (called cfg). This function then instantiates and
    executes the resulting pipeline object.
    Filters steps if `pipe_steps` is provided, otherwise executes the entire pipeline.
    For more information on hydra.main, please see
    https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/."""

    from hydra.utils import instantiate
    from omegaconf import OmegaConf, errors

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
    pipeline = instantiate(cfg, _convert_="all")
    pipeline.execute()


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


@app.command()
def download_mlflow_data(
    tracking_uri: Annotated[
        str, typer.Option(help="The tracking URI of the MLFlow server.")
    ] = "https://mlflow.intertwin.fedcloud.eu/",
    experiment_id: Annotated[
        str, typer.Option(help="The experiment ID that you wish to retrieve data from.")
    ] = "48",
    output_file: Annotated[
        str, typer.Option(help="The file path to save the data to.")
    ] = "mlflow_data.csv",
):
    """Download metrics data from MLFlow experiments and save to a CSV file.

    Requires MLFlow authentication if the server is configured to use it.
    Authentication must be provided via the following environment variables:
    'MLFLOW_TRACKING_USERNAME' and 'MLFLOW_TRACKING_PASSWORD'.
    """

    mlflow_credentials_set = (
        "MLFLOW_TRACKING_USERNAME" in os.environ and "MLFLOW_TRACKING_PASSWORD" in os.environ
    )
    if not mlflow_credentials_set:
        print(
            "\nWarning: MLFlow authentication environment variables are not set. "
            "If the server requires authentication, your request will fail."
            "You can authenticate by setting environment variables before running:\n"
            "\texport MLFLOW_TRACKING_USERNAME=your_username\n"
            "\texport MLFLOW_TRACKING_PASSWORD=your_password\n"
        )

    import mlflow
    import pandas as pd
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Handling authentication
    try:
        print(f"\nConnecting to MLFlow server at {tracking_uri}")
        print(f"Accessing experiment ID: {experiment_id}")
        runs = client.search_runs(experiment_ids=[experiment_id])
        print(f"Authentication successful! Found {len(runs)} runs.")
    except mlflow.MlflowException as e:
        status_code = e.get_http_status_code()
        if status_code == 401:
            print(
                "Authentication with MLFlow failed with code 401! Either your "
                "environment variables are not set or they are incorrect!"
            )
            return
        else:
            raise e

    all_metrics = []
    for run_idx, run in enumerate(runs):
        run_id = run.info.run_id
        metric_keys = run.data.metrics.keys()  # Get all metric names

        print(f"Processing run {run_idx+1}/{len(runs)}")
        for metric_name in metric_keys:
            metrics = client.get_metric_history(run_id, metric_name)
            for metric in metrics:
                all_metrics.append(
                    {
                        "run_id": run_id,
                        "metric_name": metric.key,
                        "value": metric.value,
                        "step": metric.step,
                        "timestamp": metric.timestamp,
                    }
                )

    if not all_metrics:
        print("No metrics found in the runs")
        return

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(output_file, index=False)
    print(f"Saved data to '{Path(output_file).resolve()}'!")


if __name__ == "__main__":
    app()
