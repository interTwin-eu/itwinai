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

    from itwinai.scalability import (
        backup_scalability_metrics,
        convert_matching_files_to_dataframe,
    )
    from itwinai.torch.monitoring.plotting import (
        calculate_average_gpu_utilization,
        calculate_total_energy_expenditure,
        gpu_bar_plot,
    )

    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        raise ValueError(f"The provided log_dir, '{log_dir_path.resolve()}', does not exist.")

    plot_dir_path = Path(plot_dir)
    if pattern.lower() == "none":
        pattern = None

    gpu_data_df = convert_matching_files_to_dataframe(pattern=pattern, log_dir=log_dir_path)

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

    backup_scalability_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        backup_dir=backup_dir,
        metric_df=gpu_data_df,
        filename="gpu_data.csv",
    )


@app.command()
def generate_communication_plot(
    log_dir: str = "scalability-metrics/communication-data",
    pattern: str = r"(.+)_(\d+)_(\d+)\.csv$",
    output_file: str = "plots/communication_plot.png",
    do_backup: bool = False,
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

    from itwinai.scalability import (
        backup_scalability_metrics,
        convert_matching_files_to_dataframe,
    )
    from itwinai.torch.profiling.communication_plot import (
        communication_overhead_stacked_bar_plot,
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

    fig, _ = communication_overhead_stacked_bar_plot(values, strategies, gpu_numbers)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path)
    print(f"\nSaved computation vs. communication plot at '{output_path.resolve()}'.")

    if not do_backup:
        return

    backup_scalability_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        backup_dir=backup_dir,
        metric_df=communication_df,
        filename="communication_data.csv",
    )


@app.command()
def generate_scalability_plot(
    pattern: str = "None",
    log_dir: str = "scalability-metrics/epoch-time",
    do_backup: bool = False,
    backup_dir: str = "backup-scalability-metrics/",
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """Creates two scalability plots from measured wall-clock times of an experiment
    run and saves them to file. Uses pattern to filter out files if given, otherwise
    it will try to use all files it finds in the given log directory. Will store all
    the data that was used as a backup file if do_backup is provided.
    """

    from itwinai.scalability import (  # archive_data,
        backup_scalability_metrics,
        convert_matching_files_to_dataframe,
        create_absolute_plot,
        create_relative_plot,
    )

    log_dir_path = Path(log_dir)
    if pattern.lower() == "none":
        pattern = None

    expected_columns = {"name", "nodes", "epoch_id", "time"}
    combined_df = convert_matching_files_to_dataframe(
        log_dir=log_dir_path, pattern=pattern, expected_columns=expected_columns
    )
    print("Merged CSV:")
    print(combined_df)

    avg_time_df = (
        combined_df.drop(columns="epoch_id").groupby(["name", "nodes"]).mean().reset_index()
    )
    print("\nAvg over name and nodes:")
    print(avg_time_df.rename(columns=dict(time="avg(time)")))

    create_absolute_plot(avg_time_df)
    create_relative_plot(avg_time_df)

    if not do_backup:
        return

    backup_scalability_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        backup_dir=backup_dir,
        metric_df=combined_df,
        filename="epoch_time.csv",
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
