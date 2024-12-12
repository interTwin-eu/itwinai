# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import subprocess
from pathlib import Path
from typing import List

from itwinai.slurm.slurm_constants import JUWELS_HPC_MODULES
from itwinai.slurm.utils import remove_indentation_from_multiline_string


class SlurmScriptBuilder:

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
        file_folder: Path = Path("slurm_scripts"),
    ):
        self.job_name = job_name
        self.account = account
        self.time = time
        self.partition = partition
        self.std_out = std_out
        self.err_out = err_out
        self.num_nodes = num_nodes
        self.num_tasks_per_node = num_tasks_per_node
        self.gpus_per_node = gpus_per_node
        self.cpus_per_gpu = cpus_per_gpu
        self.distributed_strategy = distributed_strategy
        self.python_venv = python_venv
        self.debug = debug
        self.file_folder = file_folder

        if self.cpus_per_gpu > 0:
            self.omp_num_threads = self.cpus_per_gpu
        else:
            self.omp_num_threads = 1

    def get_training_command(self) -> str:
        return "python main.py"

    def get_srun_command(self, torch_log_dir: str = "logs_torchrun"):
        if self.distributed_strategy in ["ddp", "deepspeed"]:
            rdzv_endpoint = (
                '\'$(scontrol show hostnames "$SLURM_JOB_NODELIST"'
                " | head -n 1)':29500"
            )
            main_command = rf"""
            srun --cpu-bind=none --ntasks-per-node=1 \
            bash -c "torchrun \
                --log_dir='{torch_log_dir}' \
                --nnodes=$SLURM_NNODES \
                --nproc_per_node=$SLURM_GPUS_PER_NODE \
                --rdzv_id=$SLURM_JOB_ID \
                --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
                --rdzv_backend=c10d \
                --rdzv_endpoint={rdzv_endpoint} \
                {self.get_training_command()}"
        """
        else:
            num_tasks = self.gpus_per_node * self.num_nodes

            # E.g. "0,1,2,3" with 4 GPUs per node
            cuda_visible_devices = ",".join(str(i) for i in range(self.gpus_per_node))

            main_command = rf"""
            export CUDA_VISIBLE_DEVICES="{cuda_visible_devices}"
            srun --cpu-bind=none \
                --ntasks-per-node=$SLURM_GPUS_PER_NODE \
                --cpus-per-task=$SLURM_CPUS_PER_GPU \
                --ntasks={num_tasks} \
                {self.get_training_command()}
        """

        main_command = main_command.strip()
        return remove_indentation_from_multiline_string(main_command)

    def get_debug_command(self):
        debug_print_command = rf"""
        echo "### SLURM Variables ###"
        echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
        echo "SLURM_JOB_ID: $SLURM_JOB_ID"
        echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
        echo "SLURM_NNODES: $SLURM_NNODES"
        echo "SLURM_NTASKS: $SLURM_NTASKS"
        echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
        echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
        echo "SLURMD_NODENAME: $SLURMD_NODENAME"
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        echo ""
        echo "### Other Variables ###"
        echo "Distributed Strategy: {self.distributed_strategy}"
        """
        debug_print_command = debug_print_command.strip()
        return remove_indentation_from_multiline_string(debug_print_command)

    def get_setup_command(self):
        setup_command = rf"""
            ml {" ".join(JUWELS_HPC_MODULES)}
            source {self.python_venv}/bin/activate
            export OMP_NUM_THREADS={self.omp_num_threads}
        """

        if self.debug:
            setup_command += "\n" + self.get_debug_command()

        setup_command = setup_command.strip()
        return remove_indentation_from_multiline_string(setup_command)

    def get_slurm_script(
        self, setup_command: str | None = None, main_command: str | None = None
    ):
        if setup_command is None:
            setup_command = self.get_setup_command()
        if main_command is None:
            main_command = self.get_srun_command()

        template = rf"""
            #!/bin/bash

            # Job configuration
            #SBATCH --job-name={self.job_name}
            #SBATCH --account={self.account}
            #SBATCH --partition={self.partition}
            #SBATCH --time={self.time}

            #SBATCH --output={self.std_out}
            #SBATCH --error={self.err_out}

            # Resources allocation
            #SBATCH --nodes={self.num_nodes}
            #SBATCH --ntasks-per-node={self.num_tasks_per_node}
            #SBATCH --gpus-per-node={self.gpus_per_node}
            #SBATCH --cpus-per-gpu={self.cpus_per_gpu}
            #SBATCH --exclusive

            {setup_command}

            {main_command}"""

        # Removing superfluous indentation
        template = template.strip()
        return remove_indentation_from_multiline_string(template)

    def process_slurm_script(
        self,
        setup_command: str | None = None,
        main_command: str | None = None,
        file_path: Path | None = None,
        retain_file: bool = True,
        run_script: bool = True,
    ) -> None:
        script = self.get_slurm_script(
            setup_command=setup_command, main_command=main_command
        )
        if not run_script and not retain_file:
            print(script)
            return


        if file_path is None:
            file_name = (
                f"{self.distributed_strategy}"
                f"-{self.num_nodes}x{self.gpus_per_node}.sh"
            )
            file_path = self.file_folder / file_name

        if file_path.exists():
            raise ValueError(
                f"File '{file_path.resolve()}' already exists! Give a different path "
                f"or delete the file first!"
            )

        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w") as f:
            f.write(script)

        if run_script:
            subprocess.run(["sbatch", str(file_path.resolve())])

        if not retain_file:
            file_path.unlink()

    def run_slurm_script_all_strategies(
        self,
        setup_command: str | None = None,
        main_command: str | None = None,
        file_folder: Path = Path("slurm_scripts"),
        retain_file: bool = True,
        run_script: bool = True,
        strategies: List[str] = ["ddp", "horovod", "deepspeed"],
    ):
        self.file_folder = file_folder
        for strategy in strategies:
            self.distributed_strategy = strategy
            self.process_slurm_script(
                setup_command=setup_command,
                main_command=main_command,
                retain_file=retain_file,
                run_script=run_script
            )

    def run_scaling_test(
        self,
        setup_command: str | None = None,
        main_command: str | None = None,
        file_folder: Path = Path("slurm_scripts"),
        retain_file: bool = True,
        run_script: bool = True,
        strategies: List[str] = ["ddp", "horovod", "deepspeed"],
        num_nodes_list: List[int] = [1, 2, 4, 8],
    ):
        for num_nodes in num_nodes_list:
            self.num_nodes = num_nodes
            self.run_slurm_script_all_strategies(
                setup_command=setup_command,
                main_command=main_command,
                file_folder=file_folder,
                retain_file=retain_file,
                run_script=run_script,
                strategies=strategies,
            )


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
    debug = True

    slurm_script = SlurmScriptBuilder(
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
    )

    template = slurm_script.get_slurm_script()
    print(template)


if __name__ == "__main__":
    main()
