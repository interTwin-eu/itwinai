# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import List

from itwinai.parser import ArgumentParser


def remove_indentation_from_multiline_string(multiline_string: str) -> str:
    """Removes *all* indentation from the start of each line in a multi-line string.

    If you want to remove only the shared indentation of all lines, thus preserving
    indentation for nested structures, use the builtin `textwrap.dedent` function instead.

    The main purpose of this function is allowing you to define multi-line strings that
    only appear indented in the code, thus increasing readability.
    """
    return "\n".join([line.lstrip() for line in multiline_string.split("\n")])


def scalability_nodes_list(value: str | List[int]) -> List[int]:
    """Checks that the value it receives conforms to the comma-separated integer
    constraint and returns the parsed list if successful.

    Returns:
        The list of integers that was parsed.

    Raises:
        ValueError: If unable to parse the integers e.g. due to formatting errors.
    """

    if isinstance(value, list):
        if not all(isinstance(x, int) for x in value):
            raise ValueError(f"Provided list, '{value}', contains non-integer values.")
        return value

    try:
        return [int(n) for n in value.split(",")]
    except ValueError:
        raise ValueError(
            f"Invalid input: '{value}', must be formatted as comma-separated integers."
        )


def get_slurm_job_parser() -> ArgumentParser:
    # Default arguments for the SLURM script configuration
    default_account = "intertwin"
    default_time = "00:30:00"
    default_partition = "develbooster"
    default_job_name = None
    default_std_out = None
    default_err_out = None
    default_num_nodes = 1
    default_gpus_per_node = 4
    default_cpus_per_task = 16

    # Default other arguments
    default_mode = "single"
    default_distributed_strategy = "ddp"
    default_config_file_path = "."
    default_config_file = "config"
    default_pipe_key = "rnn_training_pipeline"
    # Command to be executed before the main process starts.
    default_pre_exec_command = None
    default_training_command = None
    default_python_venv = ".venv"
    default_scalability_nodes = "1,2,4,8"
    default_profiling_sampling_rate = 10  # py-spy profiler sample rate/frequency

    parser = ArgumentParser(parser_mode="omegaconf")

    # Arguments specific to the SLURM script configuration
    parser.add_argument(
        "--job-name",
        type=str,
        default=default_job_name,
        help="The name of the SLURM job.",
    )
    parser.add_argument(
        "--account",
        type=str,
        default=default_account,
        help="The billing account for the SLURM job.",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=default_time,
        help="The time limit of the SLURM job.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=default_partition,
        help="Which partition of the cluster the SLURM job is going to run on.",
    )
    parser.add_argument(
        "--std-out", type=str, default=default_std_out, help="The standard out file."
    )
    parser.add_argument(
        "--err-out", type=str, default=default_err_out, help="The error out file."
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=default_num_nodes,
        help="The number of nodes that the SLURM job is going to run on.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=default_gpus_per_node,
        help="The requested number of GPUs per node.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=default_cpus_per_task,
        help="The requested number of CPUs per task.",
    )

    # Arguments specific to the itwinai pipeline
    parser.add_argument(
        "--config-path",
        type=str,
        default=default_config_file_path,
        help="The path to the directory containing the config file to use for training.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=default_config_file,
        help="The name of the config file to use for training.",
    )
    parser.add_argument(
        "--pipe-key",
        type=str,
        default=default_pipe_key,
        help="Which pipe key to use for running the pipeline.",
    )

    # Arguments specific to the SLURM Builder
    parser.add_argument(
        "--mode",
        choices=["scaling-test", "runall", "single"],
        default=default_mode,
        help="Which mode to run, e.g. scaling test, all strategies, or a single run.",
    )
    parser.add_argument(
        "--dist-strat",
        choices=["ddp", "horovod", "deepspeed"],
        default=default_distributed_strategy,
        help="Which distributed strategy to use.",
    )
    parser.add_argument(
        "--pre-exec-cmd",
        type=str,
        default=default_pre_exec_command,
        help="The pre-execution command to use for the python script.",
    )
    parser.add_argument(
        "--training-cmd",
        type=str,
        default=default_training_command,
        help="The training command to use for the python script.",
    )
    parser.add_argument(
        "--python-venv",
        type=str,
        default=default_python_venv,
        help="Which python venv to use for running the command.",
    )
    parser.add_argument(
        "--scalability-nodes",
        type=scalability_nodes_list,
        default=default_scalability_nodes,
        help="A comma-separated list of node numbers to use for the scalability test.",
    )
    parser.add_argument(
        "-pr",
        "--profiling-sampling-rate",
        type=int,
        default=default_profiling_sampling_rate,
        help="The rate at which the py-spy profiler should sample the call stack.",
    )

    # Boolean arguments where you only need to include the flag and not an actual value
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to include debugging information or not.",
    )
    parser.add_argument(
        "-ns",
        "--no-save-script",
        action="store_true",
        help="Whether to save the script after processing it.",
    )
    parser.add_argument(
        "-nj",
        "--no-submit-job",
        action="store_true",
        help="Whether to submit the job when processing the script.",
    )
    parser.add_argument(
        "-ps",
        "--py-spy",
        action="store_true",
        help="Whether to activate profiling with py-spy.",
    )

    return parser
