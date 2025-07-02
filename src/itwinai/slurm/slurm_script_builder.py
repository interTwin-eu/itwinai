# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

from pydantic import BaseModel

from itwinai.slurm.slurm_constants import JUWELS_HPC_MODULES, SLURM_TEMPLATE
from itwinai.slurm.utils import (
    get_slurm_job_parser,
    remove_indentation_from_multiline_string,
)

cli_logger = logging.getLogger("cli_logger")


class SlurmScriptConfiguration(BaseModel):
    """Configuration object for the SLURM script. It contains all the settings for the
    SLURM script such as which hardware you are requesting or for how long to run it.
    As it allows for any ``pre_exec_command`` and ``exec_command``, it should work for
    any SLURM script.
    """

    # Settings for the SLURM configuration
    job_name: str | None = None
    account: str
    partition: str
    time: str

    std_out: Path | None = None
    err_out: Path | None = None

    num_nodes: int
    num_tasks_per_node: int
    gpus_per_node: int
    cpus_per_task: int

    # Typically used to set up the environment before executing the command,
    # e.g. "ml Python", "source .venv/bin/activate" etc.
    pre_exec_command: str | None = None

    # Command to execute, typically an 'srun' command
    exec_command: str | None = None

    def format_script(self) -> str:
        """Uses the provided configuration parameters and formats a SLURM script with
        the requested settings.

        Returns:
            str: A string containing the SLURM script.
        """
        return SLURM_TEMPLATE.format_map(self.model_dump())


class SlurmScriptBuilder:
    """Builds a SLURM script for various itwinai use cases, typically machine-learning
    use cases. Uses the provided ``SlurmScriptConfiguration`` to build the script. Some
    methods, such as the run-all or scaling-test methods, will modify the given
    configuration in-place, so be careful if reusing the configuration object.
    """

    def __init__(
        self,
        slurm_script_configuration: SlurmScriptConfiguration,
        distributed_strategy: str,
        pre_exec_command: str | None = None,
        training_command: str | None = None,
        python_venv: str = ".venv",
        debug: bool = False,
        config_name: str = "config",
        config_path: str = ".",
        pipe_key: str = "training_pipeline",
        file_folder: Path = Path("slurm_scripts"),
        py_spy_profiling: bool = False,
        profiling_sampling_rate: int = 10,
    ):
        self.slurm_script_configuration = slurm_script_configuration
        self.distributed_strategy = distributed_strategy
        self.pre_exec_command = pre_exec_command
        self.training_command = training_command

        self.python_venv = python_venv
        self.debug = debug
        self.file_folder = file_folder

        self.py_spy_profiling = py_spy_profiling
        self.profiling_sampling_rate = profiling_sampling_rate

        # exec-pipeline-specific commands
        self.config_name = config_name
        self.config_path = config_path
        self.pipe_key = pipe_key

        self.omp_num_threads = max(
            1,
            self.slurm_script_configuration.cpus_per_task
            // self.slurm_script_configuration.gpus_per_node,
        )

    @property
    def training_cmd_formatter(self) -> Dict[str, str]:
        return {
            "dist_strat": self.distributed_strategy,
            "config_name": self.config_name,
            "config_path": self.config_path,
            "pipe_key": self.pipe_key,
        }

    def generate_identifier(self) -> str:
        num_nodes = self.slurm_script_configuration.num_nodes
        gpus_per_node = self.slurm_script_configuration.gpus_per_node
        return f"{self.distributed_strategy}-{num_nodes}x{gpus_per_node}"

    def _handle_directory_existence(self, directory: Path, should_create: bool) -> None:
        if directory.exists():
            return

        dir_name = str(directory.resolve())
        if should_create:
            directory.mkdir(parents=True)
            cli_logger.info(f"Creating directory '{dir_name}'")
        else:
            cli_logger.warning(
                "If you're submitting the SLURM job manually, make sure to create "
                f"the directory '{dir_name}' first. This is handled automatically when using "
                "the SLURM builder."
            )

    def get_training_command(self) -> str:
        if self.training_command:
            return self.training_command.format(**self.training_cmd_formatter)

        # Default for the TorchTrainer
        default_command = rf"""
            $(which itwinai) exec-pipeline \
            strategy={self.distributed_strategy} \
            checkpoints_location=checkpoints_{self.distributed_strategy}
        """
        default_command = default_command.strip()
        return remove_indentation_from_multiline_string(default_command)

    def get_pre_exec_command(self) -> str:
        """Generates a pre-execution command for the SLURM script. This will load the
        standard HPC modules, source the given python venv and set the OpenMP number
        of threads. Will also add debug echo statements if debug flag is set to True."""
        pre_exec_command = rf"""
            ml {" ".join(JUWELS_HPC_MODULES)}
            source {self.python_venv}/bin/activate
            export OMP_NUM_THREADS={self.omp_num_threads}
        """

        if self.pre_exec_command:
            pre_exec_command += f"\n{self.pre_exec_command}"
        if self.debug:
            pre_exec_command += "\n" + self.get_debug_command()

        pre_exec_command = pre_exec_command.strip()
        return remove_indentation_from_multiline_string(pre_exec_command)

    def get_srun_command(self, torch_log_dir: str = "logs_torchrun") -> str:
        """Generates an srun command for the different distributed ML frameworks using
        the internal training command. Sets up rendezvous connections etc. when
        necessary and exports necessary environment variables.
        """

        if self.distributed_strategy in ["ddp", "deepspeed"]:
            rdzv_endpoint = (
                "'$(scontrol show hostnames \"$SLURM_JOB_NODELIST\" | head -n 1)'i:29500"
            )
            bash_command = rf"""torchrun \
                --log_dir='{torch_log_dir}' \
                --nnodes=$SLURM_NNODES \
                --nproc_per_node=$SLURM_GPUS_PER_NODE \
                --rdzv_id=$SLURM_JOB_ID \
                --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
                --rdzv_backend=c10d \
                --rdzv_endpoint={rdzv_endpoint} \
                {self.get_training_command()}"""
            if self.py_spy_profiling:
                # Prepending the py-spy profiling command
                py_spy_profiling_dir = Path("py-spy-outputs")
                self._handle_directory_existence(
                    directory=py_spy_profiling_dir, should_create=True
                )
                py_spy_profiling_filename = (
                    rf"{self.distributed_strategy}_profile_\$SLURM_NODEID.txt"
                )
                py_spy_output_file = py_spy_profiling_dir / py_spy_profiling_filename

                bash_command = (
                    f"py-spy record -r {self.profiling_sampling_rate} -s "
                    f"-o {py_spy_output_file} -f raw -- " + bash_command
                )

            main_command = rf"""
            srun --cpu-bind=none --ntasks-per-node=1 \
            bash -c "{bash_command}"
        """
        else:
            gpus_per_node = self.slurm_script_configuration.gpus_per_node
            num_nodes = self.slurm_script_configuration.num_nodes
            num_tasks = gpus_per_node * num_nodes

            # E.g. "0,1,2,3" with 4 GPUs per node
            cuda_visible_devices = ",".join(str(i) for i in range(gpus_per_node))

            main_command = rf"""
            export CUDA_VISIBLE_DEVICES="{cuda_visible_devices}"
            srun --cpu-bind=none \
                --ntasks-per-node=$SLURM_GPUS_PER_NODE \
                --ntasks={num_tasks} \
                --cpus-per-task=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) \
                {self.get_training_command()}
        """

        main_command = main_command.strip()
        return remove_indentation_from_multiline_string(main_command)

    def get_debug_command(self) -> str:
        """Generates a debug command to be added to the pre-exec command."""

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
        echo "Current working directory: $(pwd)"
        echo "Which python: $(which python)"
        """
        debug_print_command = debug_print_command.strip()
        return remove_indentation_from_multiline_string(debug_print_command)

    def process_slurm_script(
        self,
        file_path: Path | None = None,
        save_script: bool = True,
        submit_slurm_job: bool = True,
    ) -> None:
        """Will generate and process a SLURM script according to specifications. Will
        run the script if ``submit_slurm_job`` is set to True, will store the SLURM script file
        if ``save_script`` is set to True. If both are false, it will simply print the
        script to the console without any further processing. Also creates the
        needed directories for the stderr and stdout for the script.

        Args:
            file_path: Where to store the file before processing the script. Also the location
                it will remain if ``save_script`` is set to True.
            save_script: Whether to keep or delete the file after finishing processing.
            submit_slurm_job: Whether to submit the script as a SLURM job.
        """
        job_identifier = self.generate_identifier()
        self.slurm_script_configuration.pre_exec_command = self.get_pre_exec_command()
        self.slurm_script_configuration.exec_command = self.get_srun_command()

        # Setting some default fields
        if self.slurm_script_configuration.job_name is None:
            self.slurm_script_configuration.job_name = job_identifier

        if self.slurm_script_configuration.std_out is None:
            std_out_path = Path("slurm_job_logs") / f"{job_identifier}.out"
            self.slurm_script_configuration.std_out = std_out_path
        if self.slurm_script_configuration.err_out is None:
            err_out_path = Path("slurm_job_logs") / f"{job_identifier}.err"
            self.slurm_script_configuration.err_out = err_out_path

        # Making sure the std out and std err directories exist
        self._handle_directory_existence(
            directory=self.slurm_script_configuration.std_out.parent,
            should_create=submit_slurm_job,
        )
        self._handle_directory_existence(
            directory=self.slurm_script_configuration.err_out.parent,
            should_create=submit_slurm_job,
        )

        # Generate the script using the given configuration
        script = self.slurm_script_configuration.format_script()
        if not submit_slurm_job and not save_script:
            upper_banner_str = f"{'#' * 20} SLURM Script Preview {'#' * 20}"
            cli_logger.info(upper_banner_str)
            cli_logger.info(script)
            cli_logger.info("#" * len(upper_banner_str))
            return

        temp_dir = None
        if save_script:
            file_path = file_path or self.file_folder / f"{job_identifier}.sh"
            if file_path.exists():
                raise FileExistsError(
                    f"File '{file_path.resolve()}' already exists! Give a different path "
                    f"or delete the file first."
                )

            self.file_folder.mkdir(exist_ok=True, parents=True)
            cli_logger.info(f"Storing SLURM script at '{file_path.resolve()}'.")
        else:
            temp_dir = TemporaryDirectory()
            file_path = Path(temp_dir.name) / f"{job_identifier}.sh"

        with open(file_path, "w") as f:
            f.write(script)

        if submit_slurm_job:
            subprocess.run(["sbatch", str(file_path.resolve())])

        if temp_dir:
            temp_dir.cleanup()

    def run_slurm_script_all_strategies(
        self,
        file_folder: Path = Path("slurm_scripts"),
        save_script: bool = True,
        submit_slurm_job: bool = True,
        strategies: Iterable[str] = ("ddp", "horovod", "deepspeed"),
    ):
        """Runs the SLURM script with all the given strategies.

        Does the same as the ``runall.sh`` script has been doing.
        """
        self.file_folder = file_folder
        for strategy in strategies:
            self.distributed_strategy = strategy
            job_identifier = self.generate_identifier()

            # Overriding job_name, std_out and err_out
            self.slurm_script_configuration.job_name = job_identifier
            std_out_path = Path("slurm_job_logs") / f"{job_identifier}.out"
            err_out_path = Path("slurm_job_logs") / f"{job_identifier}.err"
            self.slurm_script_configuration.std_out = std_out_path
            self.slurm_script_configuration.err_out = err_out_path

            self.process_slurm_script(
                save_script=save_script, submit_slurm_job=submit_slurm_job
            )

    def run_scaling_test(
        self,
        file_folder: Path = Path("slurm_scripts"),
        save_script: bool = True,
        submit_slurm_job: bool = True,
        strategies: Iterable[str] = ("ddp", "horovod", "deepspeed"),
        num_nodes_list: Iterable[int] = (1, 2, 4, 8),
    ):
        """Runs a scaling test, i.e. runs all the strategies with separate runs for each
        distinct number of nodes.
        """
        for num_nodes in num_nodes_list:
            self.slurm_script_configuration.num_nodes = num_nodes
            self.run_slurm_script_all_strategies(
                file_folder=file_folder,
                save_script=save_script,
                submit_slurm_job=submit_slurm_job,
                strategies=strategies,
            )


def generate_default_slurm_script() -> None:
    """Generates and optionally submits a default SLURM script.

    This function creates a SLURM script using the `SlurmScriptBuilder`, based on
    command-line arguments parsed from `get_slurm_job_parser()`. It sets up a
    basic SLURM configuration with common parameters like job name, account,
    requested resources, and execution commands.

    If `--no-submit-job` is provided, the script will not be submitted via `sbatch`.
    If `--no-save-script` is provided, the generated SLURM script will be deleted
    after execution.
    """
    parser = get_slurm_job_parser()
    args = parser.parse_args()

    num_tasks_per_node = 1

    slurm_script_configuration = SlurmScriptConfiguration(
        job_name=args.job_name,
        account=args.account,
        time=args.time,
        partition=args.partition,
        std_out=args.std_out,
        err_out=args.err_out,
        num_nodes=args.num_nodes,
        num_tasks_per_node=num_tasks_per_node,
        gpus_per_node=args.gpus_per_node,
        cpus_per_task=args.cpus_per_task,
    )

    slurm_script_builder = SlurmScriptBuilder(
        slurm_script_configuration=slurm_script_configuration,
        distributed_strategy=args.dist_strat,
        debug=args.debug,
        python_venv=args.python_venv,
        pre_exec_command=args.pre_exec_cmd,
        training_command=args.training_cmd,
        config_path=args.config_path,
        config_name=args.config_name,
        pipe_key=args.pipe_key,
        py_spy_profiling=args.py_spy,
        profiling_sampling_rate=args.profiling_sampling_rate,
    )

    submit_job = not args.no_submit_job
    save_script = not args.no_save_script

    mode = args.mode
    if mode == "single":
        slurm_script_builder.process_slurm_script(
            submit_slurm_job=submit_job, save_script=save_script
        )
    elif mode == "runall":
        slurm_script_builder.run_slurm_script_all_strategies(
            submit_slurm_job=submit_job, save_script=save_script
        )
    elif mode == "scaling-test":
        slurm_script_builder.run_scaling_test(
            submit_slurm_job=submit_job,
            save_script=save_script,
            num_nodes_list=args.scalability_nodes,
        )
