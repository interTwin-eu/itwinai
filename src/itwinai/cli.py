# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
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

import logging
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import List

import hydra
import typer
from omegaconf import DictConfig
from typing_extensions import Annotated

from .constants import BASE_EXP_NAME, RELATIVE_MLFLOW_PATH

app = typer.Typer(pretty_exceptions_enable=False)

py_logger = logging.getLogger(__name__)
cli_logger = logging.getLogger("cli_logger")


@app.command()
def generate_flamegraph(
    file: Annotated[str, typer.Option(help="The location of the raw profiling data.")],
    output_filename: Annotated[
        str, typer.Option(help="The filename of the resulting flamegraph.")
    ] = "flamegraph.svg",
):
    """Generates a flamegraph from the given profiling output."""
    script_filename = "flamegraph.pl"
    script_path = Path(__file__).parent / script_filename

    if not script_path.exists():
        py_logger.exception(f"Could not find '{script_filename}' at '{script_path}'")
        raise typer.Exit(code=1)

    try:
        with open(output_filename, "w") as out:
            subprocess.run(["perl", str(script_path), file], stdout=out, check=True)
        cli_logger.info(f"Flamegraph saved to '{output_filename}'")
    except FileNotFoundError:
        cli_logger.error("Perl is not installed or not in PATH.")
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as e:
        cli_logger.error(f"Flamegraph generation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def generate_py_spy_report(
    file: Annotated[str, typer.Option(help="The location of the raw profiling data.")],
    num_rows: Annotated[
        str,
        typer.Option(help="Number of rows to display. Pass 'all' to print the full table."),
    ] = "10",
    aggregate_leaf_paths: Annotated[
        bool,
        typer.Option(
            help="Whether to aggregate all unique leaf calls across different call stacks."
        ),
    ] = False,
    library_name: Annotated[
        str, typer.Option(help="Which library name to find the lowest contact point of.")
    ] = "itwinai",
):
    """Generates an aggregation of the raw py-spy profiling data, showing which leaf
    functions collected the most samples.
    """
    from tabulate import tabulate

    from itwinai.torch.profiling.py_spy_aggregation import (
        add_library_information,
        get_aggregated_paths,
        parse_num_rows,
        read_stack_traces,
    )

    try:
        parsed_num_rows: int | None = parse_num_rows(num_rows=num_rows)
    except ValueError as exception:
        raise typer.BadParameter(
            f"Failed to parse `num_rows` with value '{num_rows}'. Error:\n{str(exception)}",
            param_hint="num-rows",
        )

    file_path = Path(file)
    if not file_path.exists():
        raise typer.BadParameter(f"'{file_path.resolve()}' was not found!", param_hint="file")

    try:
        stack_traces = read_stack_traces(path=file_path)
    except ValueError as exception:
        cli_logger.error(
            f"Failed to read stack traces with following error:\n{str(exception)}"
        )
        raise typer.Exit(code=1)

    cli_logger.warning(
        "Multiprocessing calls (e.g. Dataloader subprocesses) might be counted"
        " multiple times (once per process) and thus be overrepresented. Take this into"
        " consideration when reading the results.\n"
    )

    add_library_information(stack_traces=stack_traces, library_name=library_name)
    leaf_stack_frames = [stack_frame_list[-1] for stack_frame_list in stack_traces]
    if aggregate_leaf_paths:
        leaf_stack_frames = get_aggregated_paths(stack_frames=leaf_stack_frames)

    leaf_stack_frames.sort(reverse=True)

    # Turn num_samples into percentages
    total_samples = sum(stack_frame.num_samples for stack_frame in leaf_stack_frames)
    for stack_frame in leaf_stack_frames:
        num_samples = stack_frame.num_samples
        percentage = 100 * num_samples / total_samples
        stack_frame.proportion = percentage

    filtered_leaf_stack_dicts = []
    if parsed_num_rows is not None:
        frames_to_print = leaf_stack_frames[:parsed_num_rows]
    else:
        frames_to_print = leaf_stack_frames

    for stack_frame in frames_to_print:
        stack_frame_dict = stack_frame.model_dump()

        # Creating a proportion string for the table and putting it at the end
        num_samples = stack_frame_dict.pop("num_samples")
        proportion = stack_frame_dict.pop("proportion")
        stack_frame_dict["proportion (n)"] = f"{proportion:.2f}% ({num_samples})"
        filtered_leaf_stack_dicts.append(stack_frame_dict)

    cli_logger.info(tabulate(filtered_leaf_stack_dicts, headers="keys", tablefmt="presto"))


@app.command()
def generate_scalability_report(
    tracking_uri: Annotated[
        str, typer.Option(help="The tracking URI of the MLFlow server.")
    ] = str(RELATIVE_MLFLOW_PATH),
    experiment_name: Annotated[
        str,
        typer.Option(help="The name of the mlflow experiment to use for the GPU data report."),
    ] = BASE_EXP_NAME,
    plot_dir: Annotated[
        str, typer.Option(help=("Which directory to save the resulting plots in."))
    ] = "plots",
    run_names: Annotated[
        str | None,
        typer.Option(
            help=(
                "Which run names to read, presented as comma-separated values"
                " e.g. 'run0,run1'."
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
    include_communication: Annotated[
        bool,
        typer.Option(
            help=(
                "Include communication data in the scalability report. Disclaimer:"
                " Communication fractions are unreliable and vary significantly for different"
                " HPC systems."
            )
        ),
    ] = False,
    no_warnings: Annotated[
        bool,
        typer.Option(help=("Create plots without warnings.")),
    ] = False,
):
    """Generates scalability reports for epoch time, GPU data, and communication data
    based the mlflow logs.

    This command processes runs under the given experiment at a tracking uri.
    It generates plots and metrics for scalability analysis and saves them in the `plot_dir`.
    """
    from mlflow.tracking import MlflowClient

    from itwinai.scalability_report.reports import (
        communication_data_report,
        computation_data_report,
        epoch_time_report,
        gpu_data_report,
    )
    from itwinai.utils import normalize_tracking_uri

    run_names_list = run_names.split(",") if run_names else None

    # Remove symbolic links and resolve the path
    plot_dir_path = Path(plot_dir).resolve()

    # ensure the tracking URI is normalized
    tracking_uri = normalize_tracking_uri(tracking_uri)
    mlflow_client = MlflowClient(tracking_uri=tracking_uri)

    plot_dir_path.mkdir(exist_ok=True, parents=True)

    epoch_time_table = epoch_time_report(
        plot_dir=plot_dir_path,
        mlflow_client=mlflow_client,
        experiment_name=experiment_name,
        run_names=run_names_list,
        plot_file_suffix=plot_file_suffix,
    )

    ray_footnote = None
    if not no_warnings:
        ray_footnote = (
            "For ray strategies, the number of GPUs is the number of GPUs per HPO trial.\n"
            " Keep in mind that multiple trials increase the total energy consumption."
        )

    gpu_data_table = gpu_data_report(
        plot_dir=plot_dir_path,
        mlflow_client=mlflow_client,
        experiment_name=experiment_name,
        run_names=run_names_list,
        plot_file_suffix=plot_file_suffix,
        ray_footnote=ray_footnote,
    )

    # Disclaimer for the plots
    if not no_warnings:
        typer.echo(
            typer.style("-" * 38 + "DISCLAIMER" + "-" * 38, fg=typer.colors.YELLOW, bold=True)
        )
        typer.echo(
            dedent("""
        The computation plots are experimental and do not account for parallelism.
        Calls traced by the torch profiler may overlap in time, so the sum of
        individual operation durations does not necessarily equal the total training run
        duration.

        The computed fractions are calculated as:
        (summed duration of ATen + Autograd operations) / (summed duration of all operations)

        Note:
            Different strategies handle computation and communication differently.
            Therefore, these plots should *not* be used to compare strategies solely
            based on computation fractions.

        However:
            Comparing the computation fraction across multiple GPU counts *within*
            the same strategy may provide insights into its scalability.
        """).strip()
        )
        typer.echo(typer.style("-" * 86, fg=typer.colors.YELLOW, bold=True))

    communication_data_table = None
    if include_communication:
        communication_data_table = communication_data_report(
            plot_dir=plot_dir_path,
            mlflow_client=mlflow_client,
            experiment_name=experiment_name,
            run_names=run_names_list,
            plot_file_suffix=plot_file_suffix,
        )

    computation_data_table = computation_data_report(
        plot_dir=plot_dir_path,
        mlflow_client=mlflow_client,
        experiment_name=experiment_name,
        run_names=run_names_list,
        plot_file_suffix=plot_file_suffix,
    )

    cli_logger.info("")
    if epoch_time_table is not None:
        cli_logger.info("#" * 8 + " Epoch Time Report " + "#" * 8)
        cli_logger.info(epoch_time_table + "\n")
    else:
        cli_logger.info("No Epoch Time Data Found\n")

    if gpu_data_table is not None:
        cli_logger.info("#" * 8 + " GPU Data Report " + "#" * 8)
        cli_logger.info(gpu_data_table + "\n")
    else:
        cli_logger.info("No GPU Data Found\n")

    if computation_data_table is not None:
        cli_logger.info("#" * 8 + " Computation Data Report " + "#" * 8)
        cli_logger.info(computation_data_table + "\n")
    else:
        cli_logger.info("No Computation Data Found\n")

    if include_communication:
        if communication_data_table is not None:
            cli_logger.info("#" * 8 + " Communication Data Report " + "#" * 8)
            cli_logger.info(communication_data_table + "\n")
        else:
            cli_logger.info("No Communication Data Found\n")


@app.command()
def sanity_check(
    torch: Annotated[
        bool | None, typer.Option(help=("Check also itwinai.torch modules."))
    ] = False,
    tensorflow: Annotated[
        bool | None, typer.Option(help=("Check also itwinai.tensorflow modules."))
    ] = False,
    all: Annotated[bool | None, typer.Option(help=("Check all modules."))] = False,
    optional_deps: Annotated[
        List[str] | None, typer.Option(help="List of optional dependencies.")
    ] = None,
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
def check_distributed_cluster(
    platform: Annotated[
        str, typer.Option(help=("Hardware platform: nvidia or amd"))
    ] = "nvidia",
    launcher: Annotated[
        str, typer.Option(help=("Distributed ML cluster: torchrun or ray"))
    ] = "torchrun",
):
    """This command provides a suite of tests for a quick sanity check of the network setup
    for torch distributed. Useful when working with containers on HPC.
    Remember to prepend *torchrun* in front of this command or to start a *Ray* cluster.
    """
    from itwinai.tests.distributed import test_cuda, test_gloo, test_nccl, test_ray, test_rocm

    match platform:
        case "nvidia":
            test_cuda()
        case "amd":
            test_rocm()
        case _:
            typer.echo("Unrecognized platform!")

    match launcher:
        case "torchrun":
            test_gloo()
            test_nccl()
        case "ray":
            test_ray()
        case _:
            typer.echo("Unrecognized launcher!")


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
    gpus_per_node: Annotated[
        int,
        typer.Option("--gpus-per-node", help="The requested number of GPUs per node."),
    ] = 4,
    cpus_per_task: Annotated[
        int,
        typer.Option("--cpus-per-gpu", help="The requested number of CPUs per SLURM task."),
    ] = 4,
    save_script: Annotated[
        bool,
        typer.Option(
            "--save-script",
            help="Whether to save the script or not.",
        ),
    ] = False,
    submit_job: Annotated[
        bool,
        typer.Option(
            "--submit-script",
            help="Whether to submit the script or not.",
        ),
    ] = False,
    save_dir: Annotated[
        str | None,
        typer.Option(
            "--save-dir",
            help="In which directory to save the script, if saving it.",
        ),
    ] = None,
    exec_file: Annotated[
        str | None,
        typer.Option(
            "--exec-file",
            help=(
                "The location of the file containing the execution command. Also accepts a "
                "remote url."
            ),
        ),
    ] = None,
    pre_exec_file: Annotated[
        str | None,
        typer.Option(
            "--pre-exec-file",
            help=(
                "The location of the file containing the pre-execution command. Also accepts"
                " a remote url."
            ),
        ),
    ] = None,
    use_ray: Annotated[
        bool,
        typer.Option(
            "--use-ray",
            help="Whether to enable Ray or not.",
        ),
    ] = False,
    memory: Annotated[
        str,
        typer.Option(
            "--memory",
            help="How much memory to allocate per node.",
        ),
    ] = "16G",
    exclusive: Annotated[
        bool,
        typer.Option(
            "--exclusive",
            help="Whether to make the SLURM job exclusive or not.",
        ),
    ] = False,
    run_name: Annotated[
        str,
        typer.Option(
            "--run-name",
            help="Which run name to use.",
        ),
    ] = "16G",
    exp_name: Annotated[
        str,
        typer.Option(
            "--exp-name",
            help="Which experiment name to use.",
        ),
    ] = "16G",
    container_path: Annotated[
        str | None,
        typer.Option(
            "--container-path",
            help="Path to container that should be exported.",
        ),
    ] = None,
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
    config: Annotated[
        str | None,
        typer.Option("--config", help="The path to the SLURM configuration file."),
    ] = None,
    py_spy: Annotated[
        bool, typer.Option("--py-spy", help="Whether to activate profiling with py-spy or not")
    ] = False,
    profiling_sampling_rate: Annotated[
        int,
        typer.Option(
            "--profiling-sampling-rate",
            help="The rate at which to profile with the py-spy profiler.",
        ),
    ] = 10,
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
        List[str] | None,
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
    from validators import url

    from itwinai.utils import make_config_paths_absolute, retrieve_remote_omegaconf_file

    del sys.argv[0]

    # Add current working directory to the module search path
    # so hydra will find the objects defined in the config (usually paths relative to config)
    sys.path.append(os.getcwd())

    # Process CLI arguments to handle paths
    sys.argv = make_config_paths_absolute(sys.argv)

    config = None
    if url(config_name):
        py_logger.info("Treating `config-name` as a URL.")
        config = retrieve_remote_omegaconf_file(url=config_name)

    exec_pipeline_with_compose(cfg_passthrough=config)


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def exec_pipeline_with_compose(cfg: DictConfig):
    """Hydra entry function.

    The hydra.main decorator parses a configuration file (under config_path), which contains a
    pipeline definition, and passes it to this function as an omegaconf.DictConfig object
    (called cfg).

    This function then instantiates and executes the resulting pipeline object. Filters steps
    if `pipe_steps` is provided, otherwise executes the entire pipeline. For more information
    on hydra.main, please see
    https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/.
    """

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from itwinai.utils import filter_pipeline_steps

    def range_resolver(x, y=None, step=1):
        """Custom OmegaConf resolver for range."""
        if y is None:
            return list(range(int(x)))
        return list(range(int(x), int(y), int(step)))

    if not OmegaConf.has_resolver("itwinai.range"):
        OmegaConf.register_new_resolver("itwinai.range", range_resolver)
    if not OmegaConf.has_resolver("itwinai.multiply"):
        OmegaConf.register_new_resolver("itwinai.multiply", lambda x, y: x * y)

    # Register custom OmegaConf resolver to allow dynamically computing the current working
    # directory. Example: some_field: ${itwinai.cwd:}/some/nested/path/in/current/working/dir
    if not OmegaConf.has_resolver("itwinai.cwd"):
        OmegaConf.register_new_resolver("itwinai.cwd", os.getcwd)

    pipe_steps = OmegaConf.select(cfg, "pipe_steps", default=None)
    pipe_key = OmegaConf.select(cfg, "pipe_key", default="training_pipeline")

    pipeline_cfg = OmegaConf.select(cfg, key=pipe_key)
    if pipeline_cfg is None:
        py_logger.error(
            f"Could not find pipeline key '{pipe_key}'. Make sure that you provide the full "
            "dotpath to your pipeline key."
        )
        raise typer.Exit(1)

    if pipe_steps:
        filter_pipeline_steps(pipeline_cfg=pipeline_cfg, pipe_steps=pipe_steps)
    else:
        py_logger.info("No steps selected. Executing the whole pipeline.")

    # Instantiate and execute the pipeline
    pipeline = instantiate(pipeline_cfg, _convert_="all")
    pipeline.execute()


@app.command()
def mlflow_ui(
    path: str = typer.Option("mllogs/mlflow", help="Path to logs storage."),
    port: int = typer.Option(5000, help="Port on which the MLFlow UI is listening."),
    host: str = typer.Option(
        "127.0.0.1",
        help="Which host to use. Switch to '0.0.0.0' to e.g. allow for port-forwarding.",
    ),
):
    """Visualize logs with Mlflow."""
    import subprocess

    subprocess.run(f"mlflow ui --backend-store-uri {path} --port {port} --host {host}".split())


@app.command()
def mlflow_server(
    path: str = typer.Option("mllogs/mlflow", help="Path to logs storage."),
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
        cli_logger.warning(
            "MLFlow authentication environment variables are not set. If the server requires"
            " authentication, your request will fail. You can authenticate by setting"
            " environment variables before running:\n"
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
        cli_logger.info(f"\nConnecting to MLFlow server at {tracking_uri}")
        cli_logger.info(f"Accessing experiment ID: {experiment_id}")
        runs = client.search_runs(experiment_ids=[experiment_id])
        cli_logger.info(f"Authentication successful! Found {len(runs)} runs.")
    except mlflow.MlflowException as e:
        status_code = e.get_http_status_code()
        if status_code == 401:
            cli_logger.error(
                "Authentication with MLFlow failed with code 401! Either your "
                "environment variables are not set or they are incorrect!"
            )
            raise typer.Exit(code=1)
        else:
            cli_logger.info(e.message)
            raise typer.Exit(code=1)

    all_metrics = []
    for run_idx, run in enumerate(runs):
        run_id = run.info.run_id
        metric_keys = run.data.metrics.keys()  # Get all metric names

        cli_logger.info(f"Processing run {run_idx + 1}/{len(runs)}")
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
        cli_logger.error("No metrics found in the runs")
        raise typer.Exit(code=1)

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(output_file, index=False)
    cli_logger.info(f"Saved data to '{Path(output_file).resolve()}'!")


@app.command()
def tensorboard_ui(
    path: str = typer.Option("mllogs/tensorboard", help="Path to logs storage."),
    port: int = typer.Option(6006, help="Port on which the Tensorboard UI is listening."),
    host: str = typer.Option(
        "127.0.0.1",
        help="Which host to use. Switch to '0.0.0.0' to e.g. allow for port-forwarding.",
    ),
):
    """Visualize logs with TensorBoard."""
    import subprocess

    subprocess.run(f"tensorboard --logdir={path} --port={port} --host={host}".split())


if __name__ == "__main__":
    app()
