# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from itwinai.slurm.create_slurm import (
    SlurmScriptBuilder,
    remove_indentation_from_multiline_string,
)


class EuracSlurmScriptBuilder(SlurmScriptBuilder):

    def __init__(
        self,
        job_name: str,
        account: str,
        time: str,
        partition: str,
        std_out: str,
        err_out: str,
        num_nodes: int,
        num_tasks_per_node: int,
        gpus_per_node: int,
        cpus_per_gpu: int,
        distributed_strategy: str,
        python_venv: str = ".venv",
        debug: bool = False,
        config_file: str = "config.yaml",
        pipe_key: str = "rnn_training_pipeline",
    ):
        super().__init__(
            job_name,
            account,
            time,
            partition,
            std_out,
            err_out,
            num_nodes,
            num_tasks_per_node,
            gpus_per_node,
            cpus_per_gpu,
            distributed_strategy,
            python_venv,
            debug,
        )
        self.config_file = config_file
        self.pipe_key = pipe_key

    def get_training_command(self):
        training_command = rf"""
        $(which itwinai) exec-pipeline \
            --config {self.config_file} \
            --pipe-key {self.pipe_key} \
            -o strategy={self.distributed_strategy}
        """
        training_command = training_command.strip()
        return remove_indentation_from_multiline_string(training_command)


def parse_args():
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
    default_config_file = "config.yaml"
    default_pipe_key = "rnn_training_pipeline"

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
        "--config-file",
        type=str,
        default=default_config_file,
        help="Which config file to use for training.",
    )
    parser.add_argument(
        "--pipe-key",
        type=str,
        default=default_pipe_key,
        help="Which pipe key to use for running the pipeline.",
    )
    parser.add_argument(
        "--dist-strat",
        choices=["ddp", "horovod", "deepspeed"],
        default=default_pipe_key,
        help="Which distributed strategy to use.",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Whether to include debugging information or not",
    )

    return parser.parse_args()  

def single_run(args): 
    # SLURM args
    job_name = args.job_name
    account = args.account
    time = args.time
    partition = args.partition
    std_out = args.std_out
    err_out = args.err_out
    num_nodes = args.num_nodes
    num_tasks_per_node = args.num_tasks_per_node
    gpus_per_node = args.gpus_per_node
    cpus_per_gpu = args.cpus_per_gpu

    # Other args
    config_file = args.config_file
    pipe_key = args.pipe_key
    distributed_strategy = args.dist_strat
    debug = args.debug

    slurm_script = EuracSlurmScriptBuilder(
        # TODO: Move the args here instead of storing them to variables
        job_name=job_name,
        account=account,
        time=time,
        partition=partition,
        std_out=std_out,
        err_out=err_out,
        num_nodes=num_nodes,
        num_tasks_per_node=num_tasks_per_node,
        gpus_per_node=gpus_per_node,
        cpus_per_gpu=cpus_per_gpu,
        distributed_strategy=distributed_strategy,
        debug=debug,
        config_file=config_file,
        pipe_key=pipe_key,
    )

    file_path = Path(f"slurm_scripts/{distributed_strategy}-test.sh")
    file_path.parent.mkdir(exist_ok=True, parents=True)
    # slurm_script.run_slurm_script(file_path=file_path, retain_file=)

def run_all(args): 
    pass

def scalability_test(args): 
    pass


def main():
    args = parse_args()

    mode = args.mode
    if mode == "single": 
        single_run(args)
    elif mode == "runall": 
        run_all(args)
    elif mode == "scaling-test":
        scalability_test(args)
    else: 
        # This shouldn't really ever happen, but checking just in case
        raise ValueError(
            f"Mode can only be 'single', 'runall' or 'scaling-test', but was '{mode}'." 
        )

    # SLURM args
    job_name = args.job_name
    account = args.account
    time = args.time
    partition = args.partition
    std_out = args.std_out
    err_out = args.err_out
    num_nodes = args.num_nodes
    num_tasks_per_node = args.num_tasks_per_node
    gpus_per_node = args.gpus_per_node
    cpus_per_gpu = args.cpus_per_gpu

    # Other args
    config_file = args.config_file
    pipe_key = args.pipe_key

    dist_strats = ["ddp"]  # , "deepspeed", "horovod"]
    for distributed_strategy in dist_strats:
        slurm_script = EuracSlurmScriptBuilder(
            job_name=job_name,
            account=account,
            time=time,
            partition=partition,
            std_out=std_out,
            err_out=err_out,
            num_nodes=num_nodes,
            num_tasks_per_node=num_tasks_per_node,
            gpus_per_node=gpus_per_node,
            cpus_per_gpu=cpus_per_gpu,
            distributed_strategy=distributed_strategy,
            debug=True,
            config_file=config_file,
            pipe_key=pipe_key,
        )

        file_path = Path(f"slurm_scripts/{distributed_strategy}-test.sh")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        # slurm_script.run_slurm_script(file_path=file_path, retain_file=)
        script = slurm_script.get_slurm_script()
        with open("my_script.sh", "w") as f:
            f.write(script)


if __name__ == "__main__":
    main()
