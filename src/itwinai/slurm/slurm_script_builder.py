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
from tempfile import NamedTemporaryFile
from typing import Dict

import typer
from validators import url

from itwinai.slurm.slurm_constants import DEFAULT_SLURM_LOG_DIR, DEFAULT_SLURM_SAVE_DIR
from itwinai.slurm.slurm_script_configuration import SlurmScriptConfiguration
from itwinai.slurm.utils import (
    get_slurm_job_parser,
    remove_indentation_from_multiline_string,
    retrieve_remote_file,
)

cli_logger = logging.getLogger("cli_logger")


class SlurmScriptBuilder:
    def __init__(
        self,
        slurm_script_configuration: SlurmScriptConfiguration,
        should_submit: bool,
        should_save: bool,
        save_path: str | Path | None = None,
    ):
        self.slurm_script_configuration = slurm_script_configuration
        self.should_submit = should_submit
        self.should_save = should_save

        if isinstance(save_path, str):
            save_path = Path(save_path)
        self.save_path = save_path

    @staticmethod
    def submit_script(script: str) -> None:
        """Submits the given script with 'sbatch' using a temporary file."""
        with NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write(script)
            # Making sure the script is written to file before the cmd is launched
            temp_file.flush()
            try:
                subprocess.run(["sbatch", temp_file.name], check=True)
            except FileNotFoundError as e:
                cli_logger.error(
                    "'sbatch' failed. Are you sure you have it installed on your system? Error"
                    f" message:\n{e}."
                )
                raise typer.Exit(1)
            except subprocess.CalledProcessError as e:
                cli_logger.error(f"'sbatch' failed to submit script. Error message:\n{e}")
                raise typer.Exit(1)

    @staticmethod
    def save_script(script: str, file_path: Path) -> None:
        """Saves the given script to the given file path."""
        if file_path.exists():
            cli_logger.error(
                f"File '{file_path.resolve()}' already exists! Give a different path or "
                "delete the file first."
            )
            raise typer.Exit(1)
        if not file_path.parent.exists():
            cli_logger.info(f"Creating directory '{file_path.parent.resolve()}'!")
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(script)
        cli_logger.info(f"Saved SLURM script to '{file_path.resolve()}'.")

    @staticmethod
    def print_script(script: str) -> None:
        """Prints the given script to stdout using the cli_logger."""
        upper_banner_str = f"{'#' * 20} SLURM Script Preview {'#' * 20}"
        cli_logger.info(upper_banner_str)
        cli_logger.info(script)
        cli_logger.info("#" * len(upper_banner_str))

    def _set_default_config_fields(self) -> None:
        """Sets the job_name, std_out and err_out fields to default values if they are missing
        from the configuration object.
        """
        if self.slurm_script_configuration.job_name is None:
            self.slurm_script_configuration.job_name = "job"

        job_name = self.slurm_script_configuration.job_name
        if self.slurm_script_configuration.std_out is None:
            std_out_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_name}.out"
            self.slurm_script_configuration.std_out = std_out_path

        if self.slurm_script_configuration.err_out is None:
            std_err_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_name}.err"
            self.slurm_script_configuration.err_out = std_err_path

    def _ensure_log_dirs_exist(self):
        if self.slurm_script_configuration.std_out is None:
            cli_logger.error(
                "Failed to ensure existence of logging directories. Cannot ensure existence of"
                "std_out as it is None!"
            )
            raise typer.Exit(1)
        if self.slurm_script_configuration.err_out is None:
            cli_logger.error(
                "Failed to ensure existence of logging directories. Cannot ensure existence of"
                "err_out (stderr) as it is None!"
            )
            raise typer.Exit(1)

        self.slurm_script_configuration.std_out.parent.mkdir(exist_ok=True, parents=True)
        self.slurm_script_configuration.err_out.parent.mkdir(exist_ok=True, parents=True)

    def process_script(self) -> None:
        """Processes the given script by submitting and/or saving it, or by printing it to
        stdout. Also prepares the script by inserting default values wherever they are not
        set, as well as creating the needed directories.
        """
        self._set_default_config_fields()
        script = self.slurm_script_configuration.generate_script()
        if not self.should_submit and not self.should_save:
            self.print_script(script=script)
            return

        if self.should_save:
            default_save_path = (
                Path(DEFAULT_SLURM_SAVE_DIR)
                / f"{self.slurm_script_configuration.job_name}.slurm"
            )
            file_path = self.save_path if self.save_path is not None else default_save_path
            self.save_script(script=script, file_path=file_path)

        if self.should_submit:
            self._ensure_log_dirs_exist()
            self.submit_script(script=script)


class MLSlurmBuilder(SlurmScriptBuilder):
    """Builds a SLURM script tailed to distributed machine learning. Uses the provided
    ``SlurmScriptConfiguration`` to build the script and inserts values as needed.

    Note:
        The given configuration object might be modified by some of the methods.
    """

    def __init__(
        self,
        slurm_script_configuration: SlurmScriptConfiguration,
        should_submit: bool,
        should_save: bool,
        distributed_strategy: str,
        save_path: str | Path | None = None,
        pre_exec_script_location: str | Path | None = None,
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
        super().__init__(
            slurm_script_configuration=slurm_script_configuration,
            should_submit=should_submit,
            should_save=should_save,
            save_path=save_path,
        )

        self.distributed_strategy = distributed_strategy
        self.pre_exec_script_location = pre_exec_script_location
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
    def _training_cmd_dict_formatter(self) -> Dict[str, str]:
        return {
            "dist_strat": self.distributed_strategy,
            "config_name": self.config_name,
            "config_path": self.config_path,
            "pipe_key": self.pipe_key,
        }

    def _generate_job_identifier(self) -> str:
        num_nodes = self.slurm_script_configuration.num_nodes
        gpus_per_node = self.slurm_script_configuration.gpus_per_node
        return f"{self.distributed_strategy}-{num_nodes}x{gpus_per_node}"

    def get_training_command(self) -> str:
        if self.training_command:
            return self.training_command.format(**self._training_cmd_dict_formatter)

        # Default for the TorchTrainer
        default_command = rf"""
            $(which itwinai) exec-pipeline strategy={self.distributed_strategy}
        """
        default_command = default_command.strip()
        return remove_indentation_from_multiline_string(default_command)

    def get_torchrun_command(self) -> str:
        torchrun_log_dir = "torchrun_logs"
        torchrun_cmd = rf"""torchrun_jsc \
            --log_dir={torchrun_log_dir} \
            --nnodes="$SLURM_NNODES" \
            --nproc_per_node="$SLURM_GPUS_PER_NODE" \
            --rdzv_id="$SLURM_JOB_ID" \
            --rdzv_conf=is_host=$(( SLURM_NODEID == 0 ? 1 : 0 )) \
            --rdzv_backend=c10d \
            --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
            {self.get_training_command()}"""
        return remove_indentation_from_multiline_string(torchrun_cmd)

    def get_horovod_command(self) -> str:
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
        return main_command

    def get_exec_command(self) -> str:
        if self.slurm_script_configuration.exec_command is not None:
            return self.slurm_script_configuration.exec_command

        if self.distributed_strategy in ("ddp", "deepspeed"):
            tasks_per_node = "1"
            task_cmd = self.get_torchrun_command()
        else:
            tasks_per_node = "SLURM_GPUS_PER_NODE"
            task_cmd = self.get_horovod_command()

        exec_command = rf"""
            srun --cpu-bind=none --ntasks-per-node={tasks_per_node} \
            bash -c '{task_cmd}'
        """
        return remove_indentation_from_multiline_string(multiline_string=exec_command)

    def get_pre_exec_command(self) -> str:
        """Generates a pre-execution command for the SLURM script. This will load the
        standard HPC modules, source the given python venv and set the OpenMP number
        of threads. Will also add debug echo statements if debug flag is set to True."""
        pre_exec_command = ""

        if self.pre_exec_script_location:
            if url(self.pre_exec_script_location):
                cli_logger.info("Reading pre-execution script from remote url!")
                pre_exec_command = retrieve_remote_file(str(self.pre_exec_script_location))
            else:
                try:
                    with open(self.pre_exec_script_location) as file:
                        pre_exec_command = file.read()
                except FileNotFoundError:
                    cli_logger.error(
                        f"Failed to open pre-execution script. Couldn't find file: "
                        f"'{self.pre_exec_script_location}'!"
                    )
                    raise typer.Exit(1)

        if (
            "export" not in pre_exec_command
            or "MASTER_ADDR" not in pre_exec_command
            or "MASTER_PORT" not in pre_exec_command
        ):
            cli_logger.warning(
                "It seems you are not exporting MASTER_ADDR and MASTER_PORT in your pre-exec"
                " command. torchrun will not work without them!"
            )

        pre_exec_command += f"source {self.python_venv}/bin/activate"
        pre_exec_command = pre_exec_command.strip()
        return remove_indentation_from_multiline_string(pre_exec_command)

    def _set_default_config_fields(self) -> None:
        if self.slurm_script_configuration.job_name is None:
            self.slurm_script_configuration.job_name = self._generate_job_identifier()

        return super()._set_default_config_fields()

    def process_script(self) -> None:
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
        self.slurm_script_configuration.pre_exec_command = self.get_pre_exec_command()
        self.slurm_script_configuration.exec_command = self.get_exec_command()
        self._set_default_config_fields()

        # Generate the script using the given configuration
        super().process_script()

    def process_all_strategies(
        self,
        strategies: Iterable[str] = ("ddp", "horovod", "deepspeed"),
    ):
        """Runs the SLURM script with all the given strategies. Meant to replace the
        ``runall.sh`` script.
        """
        for strategy in strategies:
            self.distributed_strategy = strategy
            job_identifier = self._generate_job_identifier()

            # Updating job_name, std_out and err_out
            self.slurm_script_configuration.job_name = job_identifier
            std_out_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_identifier}.out"
            err_out_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_identifier}.err"
            self.slurm_script_configuration.std_out = std_out_path
            self.slurm_script_configuration.err_out = err_out_path

            self.process_script()

    def process_scaling_test(
        self,
        strategies: Iterable[str] = ("ddp", "horovod", "deepspeed"),
        num_nodes_list: Iterable[int] = (1, 2, 4, 8),
    ):
        """Runs a scaling test, i.e. runs all the strategies with separate runs for each
        distinct number of nodes.
        """
        for num_nodes in num_nodes_list:
            self.slurm_script_configuration.num_nodes = num_nodes
            self.process_all_strategies(strategies=strategies)


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

    slurm_script_builder = MLSlurmBuilder(
        slurm_script_configuration=slurm_script_configuration,
        should_submit=args.submit_job,
        should_save=args.save_script,
        distributed_strategy=args.dist_strat,
        save_path=args.save_path,
        debug=args.debug,
        python_venv=args.python_venv,
        pre_exec_script_location=args.pre_exec_script_location,
        training_command=args.training_cmd,
        config_path=args.config_path,
        config_name=args.config_name,
        pipe_key=args.pipe_key,
        py_spy_profiling=args.py_spy,
        profiling_sampling_rate=args.profiling_sampling_rate,
    )

    mode = args.mode
    if mode == "single":
        slurm_script_builder.process_script()
    elif mode == "runall":
        slurm_script_builder.process_all_strategies()
    elif mode == "scaling-test":
        slurm_script_builder.process_scaling_test(num_nodes_list=args.scalability_nodes)
