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
from typing import Dict, Literal

import typer
from requests.exceptions import RequestException
from validators import url

from itwinai.slurm.slurm_constants import (
    DEFAULT_PY_SPY_DIR,
    DEFAULT_SLURM_LOG_DIR,
    DEFAULT_SLURM_SAVE_DIR,
)
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
        save_dir: str | Path | None = None,
        pre_exec_file: str | None = None,
        exec_file: str | None = None,
    ):
        self.slurm_script_configuration = slurm_script_configuration
        self.should_submit = should_submit
        self.should_save = should_save
        self.pre_exec_file = pre_exec_file
        self.exec_file = exec_file

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        self.save_dir = save_dir

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
        if not file_path.parent.exists():
            cli_logger.info(f"Creating directory '{file_path.parent.resolve()}'!")
            file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "x") as f:
                f.write(script)
            cli_logger.info(f"Saved SLURM script to '{file_path.resolve()}'.")
        except FileExistsError:
            cli_logger.error(
                f"File '{file_path.resolve()}' already exists! Give a different path or "
                "delete the file first."
            )
            raise typer.Exit(1)

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
                " std_out as it is None!"
            )
            raise typer.Exit(1)
        if self.slurm_script_configuration.err_out is None:
            cli_logger.error(
                "Failed to ensure existence of logging directories. Cannot ensure existence of"
                " err_out (stderr) as it is None!"
            )
            raise typer.Exit(1)

        self.slurm_script_configuration.std_out.parent.mkdir(exist_ok=True, parents=True)
        self.slurm_script_configuration.err_out.parent.mkdir(exist_ok=True, parents=True)

    def _set_pre_exec_cmd(self) -> None:
        """Sets the pre-execution command based on the given arguments. If the passed file is
        a url then it downloads its contents. If it's a file, then it tries to read it from
        the disk.
        """
        if self.pre_exec_file is None:
            return
        if self.slurm_script_configuration.pre_exec_command is not None:
            return

        if url(self.pre_exec_file):
            try:
                contents = retrieve_remote_file(str(self.pre_exec_file))
            except RequestException as e:
                cli_logger.error(
                    f"Retrieving file from remote, '{self.pre_exec_file}', failed! Error:\n{e}"
                )
                raise typer.Exit(1)
        else:
            try:
                with open(self.pre_exec_file) as file:
                    contents = file.read()
            except FileNotFoundError:
                cli_logger.error(
                    f"Failed to open pre-execution script. Couldn't find file: "
                    f"'{self.pre_exec_file}'!"
                )
                raise typer.Exit(1)
        self.slurm_script_configuration.pre_exec_command = contents

    def _set_exec_cmd(self):
        """Sets the execution command based on the given arguments. If the passed file is
        a url then it downloads its contents. If it's a file, then it tries to read it from
        the disk.
        """
        if self.exec_file is None:
            return
        if self.slurm_script_configuration.exec_command is not None:
            return

        if url(self.exec_file):
            try:
                contents = retrieve_remote_file(str(self.exec_file))
            except RequestException as e:
                cli_logger.error(
                    f"Retrieving file from remote, '{self.exec_file}', failed! Error:\n{e}"
                )
                raise typer.Exit(1)
        else:
            try:
                with open(self.exec_file) as file:
                    contents = file.read()
            except FileNotFoundError:
                cli_logger.error(
                    f"Failed to open execution script. Couldn't find file: '{self.exec_file}'!"
                )
                raise typer.Exit(1)
        self.slurm_script_configuration.exec_command = contents

    def process_script(self) -> None:
        """Processes the given script by submitting and/or saving it, or by printing it to
        stdout. Also prepares the script by inserting default values wherever they are not
        set, as well as creating the needed directories.
        """
        self._set_default_config_fields()
        self._set_pre_exec_cmd()
        self._set_exec_cmd()
        script = self.slurm_script_configuration.generate_script()
        if not self.should_submit and not self.should_save:
            self.print_script(script=script)
            return

        if self.should_save:
            save_dir = self.save_dir if self.save_dir else Path(DEFAULT_SLURM_SAVE_DIR)
            save_path = save_dir / f"{self.slurm_script_configuration.job_name}.slurm"
            save_path = save_path.resolve()
            self.save_script(script=script, file_path=save_path)

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
        distributed_strategy: Literal["ddp", "deepspeed", "horovod"],
        use_ray: bool = False,
        pre_exec_file: str | None = None,
        exec_file: str | None = None,
        save_dir: str | Path | None = None,
        container_path: str | Path | None = None,
        training_command: str | None = None,
        python_venv: str | None = None,
        config_name: str = "config",
        config_path: str = ".",
        pipe_key: str = "training_pipeline",
        py_spy_profiling: bool = False,
        profiling_sampling_rate: int = 10,
        experiment_name: str = "main_experiment",
        run_name: str = "run1"
    ):
        super().__init__(
            slurm_script_configuration=slurm_script_configuration,
            should_submit=should_submit,
            should_save=should_save,
            save_dir=save_dir,
            pre_exec_file=pre_exec_file,
            exec_file=exec_file,
        )

        self.distributed_strategy = distributed_strategy
        self.training_command = training_command
        self.use_ray = use_ray
        self.container_path = container_path

        self.python_venv = python_venv

        self.py_spy_profiling = py_spy_profiling
        self.profiling_sampling_rate = profiling_sampling_rate

        # exec-pipeline-specific commands
        self.config_name = config_name
        self.config_path = config_path
        self.pipe_key = pipe_key

        self.experiment_name = experiment_name
        self.run_name = run_name

    def _get_training_cmd_args(self) -> Dict[str, str]:
        return {
            "dist_strat": self.distributed_strategy,
            "config_name": self.config_name,
            "config_path": self.config_path,
            "pipe_key": self.pipe_key,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
        }

    def _generate_job_identifier(self) -> str:
        num_nodes = self.slurm_script_configuration.num_nodes
        gpus_per_node = self.slurm_script_configuration.gpus_per_node
        identifier = (
            f"{'ray-' if self.use_ray else ''}"
            f"{self.distributed_strategy}-{num_nodes}x{gpus_per_node}"
        )
        return identifier

    def get_training_command(self) -> str:
        if self.training_command:
            return self.training_command.format(**self._get_training_cmd_args())

        if self.python_venv:
            itwinai_launcher = (Path(self.python_venv) / "bin" / "itwinai").resolve()
        else:
            itwinai_launcher = "itwinai"

        default_command = rf"""
            {itwinai_launcher} exec-pipeline \
            --config-name={self.config_name} \
            --config-path={self.config_path} \
            +pipe_key={self.pipe_key} \
            strategy={self.distributed_strategy} \
            experiment_name={self.experiment_name} \
            run_name={self.run_name}
        """
        default_command = default_command.strip()
        return remove_indentation_from_multiline_string(default_command)

    def get_exec_command(self) -> str:
        """Generates an execution command for the SLURM script. Considers whether ray is
        enabled or not and finds the appropriate expected bash function.
        """
        if self.slurm_script_configuration.exec_command is not None:
            return self.slurm_script_configuration.exec_command

        training_command = self.get_training_command()
        if self.distributed_strategy == "horovod" and self.use_ray:
            cli_logger.error("Horovod together with Ray is not supported!")
            raise typer.Exit(1)
        if self.use_ray and self.py_spy_profiling:
            cli_logger.error("Ray together with py-spy profiling is not supported!")
            raise typer.Exit(1)
        if self.py_spy_profiling and self.distributed_strategy == "horovod":
            cli_logger.error("Horovod together with py-spy profiling is not supported!")
            raise typer.Exit(1)

        if self.use_ray:
            base_cmd = "ray-launcher"
        elif self.distributed_strategy == "horovod":
            base_cmd = "srun-launcher"
        elif self.distributed_strategy in ("ddp", "deepspeed"):
            base_cmd = "torchrun-launcher"
        else:
            cli_logger.error(f"Invalid strategy chosen: {self.distributed_strategy}")
            raise typer.Exit(1)

        if self.container_path:
            base_cmd += "-container"

        if self.py_spy_profiling and self.distributed_strategy in ("ddp", "deepspeed"):
            py_spy_profiling_filename = (
                rf"{self.distributed_strategy}_profile_\$SLURM_NODEID.txt"
            )
            py_spy_output_file = Path(DEFAULT_PY_SPY_DIR) / py_spy_profiling_filename
            base_cmd = (
                f"py-spy-{base_cmd} '{self.profiling_sampling_rate}' '{py_spy_output_file}'"
            )

        base_cmd += f" '{training_command}'"
        return base_cmd

    def get_pre_exec_command(self) -> str:
        """Generates a pre-execution command for the SLURM script. Adds a command to source
        the python venv if given and a command to export a container path variable if given.
        """
        pre_exec_command = ""
        if self.slurm_script_configuration.pre_exec_command is not None:
            pre_exec_command = self.slurm_script_configuration.pre_exec_command

        if self.python_venv is not None:
            pre_exec_command += f"\nsource {self.python_venv}/bin/activate"

        if self.container_path is not None:
            pre_exec_command += f"\nexport CONTAINER_PATH={self.container_path}"

        pre_exec_command = pre_exec_command.strip()
        return pre_exec_command

    def _set_default_config_fields(self) -> None:
        if self.slurm_script_configuration.job_name is None:
            self.slurm_script_configuration.job_name = self._generate_job_identifier()

        super()._set_default_config_fields()

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
        self._set_pre_exec_cmd()
        self._set_exec_cmd()
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
        original_config = self.slurm_script_configuration.model_copy(deep=True)
        original_run_name = self.run_name
        for strategy in strategies:
            if strategy == "horovod" and self.use_ray:
                continue
            self.distributed_strategy = strategy
            job_identifier = self._generate_job_identifier()

            self.slurm_script_configuration = original_config.model_copy(deep=True)
            self.run_name = f"{job_identifier}-{original_run_name}"

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
        original_config = self.slurm_script_configuration.model_copy(deep=True)
        original_run_name = self.run_name
        for num_nodes in num_nodes_list:
            self.slurm_script_configuration = original_config.model_copy(deep=True)
            self.run_name = original_run_name
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

    # For all our purposes we want number of tasks to be one (in the sbatch directives)
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
        memory=args.memory,
        exclusive=args.exclusive,
    )

    slurm_script_builder = MLSlurmBuilder(
        slurm_script_configuration=slurm_script_configuration,
        should_submit=args.submit_job,
        should_save=args.save_script,
        use_ray=args.use_ray,
        container_path=args.container_path,
        distributed_strategy=args.dist_strat,
        exec_file=args.exec_file,
        pre_exec_file=args.pre_exec_file,
        save_dir=args.save_dir,
        python_venv=args.python_venv,
        training_command=args.training_cmd,
        config_path=args.config_path,
        config_name=args.config_name,
        pipe_key=args.pipe_key,
        py_spy_profiling=args.py_spy,
        profiling_sampling_rate=args.profiling_sampling_rate,
        experiment_name=args.exp_name,
        run_name=args.run_name
    )

    mode = args.mode
    if mode == "single":
        slurm_script_builder.process_script()
    elif mode == "runall":
        slurm_script_builder.process_all_strategies()
    elif mode == "scaling-test":
        slurm_script_builder.process_scaling_test(num_nodes_list=args.scalability_nodes)
