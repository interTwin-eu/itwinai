# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from argparse import ArgumentParser


def remove_indentation_from_multiline_string(multiline_string: str) -> str:
    """Removes the indentation from the start of each line in a multi-line string. The
    main purpose of this function is allowing you to define multi-line strings that
    don't touch the left margin of the editor, thus increasing readability.
    """
    return "\n".join([line.lstrip() for line in multiline_string.split("\n")])


def get_slurm_script_parser() -> ArgumentParser:
    # Default SLURM arguments
    default_job_name = "my_test_job"
    default_account = "intertwin"
    default_time = "00:01:00"
    default_partition = "develbooster"
    default_std_out = "job.out"
    default_err_out = "job.err"
    default_num_nodes = 1
    default_num_tasks_per_node = 1
    default_gpus_per_node = 4
    default_cpus_per_gpu = 4

    # Default other arguments
    default_mode = "single"
    default_distributed_strategy = "ddp"

    parser = ArgumentParser()
    parser.add_argument(
        "--job_name",
        type=str,
        default=default_job_name,
        help="The name of the SLURM job",
    )
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
        type=str,
        default=default_num_nodes,
        help="The number of nodes that the SLURM job is going to run on.",
    )
    parser.add_argument(
        "--num-tasks-per-node",
        type=str,
        default=default_num_tasks_per_node,
        help="The number of tasks per node.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=str,
        default=default_gpus_per_node,
        help="The requested number of GPUs per node.",
    )
    parser.add_argument(
        "--cpus-per-gpu",
        type=str,
        default=default_cpus_per_gpu,
        help="The requested number of CPUs per node.",
    )

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
        "--debug",
        type=bool,
        default=False,
        help="Whether to include debugging information or not",
    )

    return parser
