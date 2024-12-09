# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from create_slurm import remove_indentation_from_multiline_string, SlurmScript
from slurm_constants import JUWELS_HPC_MODULES
from typing import Literal


class EuracSlurmScript(SlurmScript):

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
        pipe_key: str = "rnn_training_pipeline"
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


def main():
    # SLURM specific settings
    job_name = "my_test_job"
    account = "intertwin"
    time = "00:01:00"
    partition = "develbooster"
    std_out = "job.out"
    err_out = "job.err"
    num_nodes = 1
    num_tasks_per_node = 1
    gpus_per_node = 4
    cpus_per_gpu = 4

    # Other settings
    distributed_strategy = "horovod"

    # EURAC specific settings
    config_file = "config.yaml"
    pipe_key = "rnn_training_pipeline"

    dist_strats = ["ddp", "deepspeed", "horovod"]
    for distributed_strategy in dist_strats:
        slurm_script = EuracSlurmScript(
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
            pipe_key=pipe_key
        )

        template = slurm_script.get_slurm_script()
        print("\n\n\n\n")
        print(template)
        print("\n\n\n\n")


if __name__ == "__main__":
    main()
