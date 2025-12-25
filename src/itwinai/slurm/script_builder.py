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

import typer
from requests.exceptions import RequestException
from validators import url

from itwinai.slurm.configuration import MLSlurmBuilderConfig, SlurmScriptConfiguration
from itwinai.slurm.constants import (
    DEFAULT_PY_SPY_DIR,
    DEFAULT_SLURM_LOG_DIR,
)
from itwinai.slurm.utils import retrieve_remote_file

cli_logger = logging.getLogger("cli_logger")

class SlurmScriptBuilder:
    """Base builder for SLURM scripts that handles defaults, execution prep, and dispatch.

    Args:
        config (SlurmScriptConfiguration): configuration object.

    Note:
        The provided ``SlurmScriptConfiguration`` may be modified while preparing the script.
    """

    config: SlurmScriptConfiguration

    def __init__(self, config: SlurmScriptConfiguration):
        self.config = config

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
        if self.config.job_name is None:
            self.config.job_name = "job"

        job_name = self.config.job_name
        if self.config.std_out is None:
            std_out_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_name}.out"
            self.config.std_out = std_out_path

        if self.config.err_out is None:
            std_err_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_name}.err"
            self.config.err_out = std_err_path

    def _ensure_log_dirs_exist(self):
        if self.config.std_out is None:
            cli_logger.error(
                "Failed to ensure existence of logging directories. Cannot ensure existence of"
                " std_out as it is None!"
            )
            raise typer.Exit(1)
        if self.config.err_out is None:
            cli_logger.error(
                "Failed to ensure existence of logging directories. Cannot ensure existence of"
                " err_out (stderr) as it is None!"
            )
            raise typer.Exit(1)

        self.config.std_out.parent.mkdir(exist_ok=True, parents=True)
        self.config.err_out.parent.mkdir(exist_ok=True, parents=True)

    def _set_pre_exec_cmd(self) -> None:
        """Sets the pre-execution command based on the given arguments. If the passed file is
        a url then it downloads its contents. If it's a file, then it tries to read it from
        the disk.
        """
        pre_exec_file = self.config.pre_exec_file
        if pre_exec_file is None:
            return
        if self.config.pre_exec_command is not None:
            return
        pre_exec_file = str(pre_exec_file)

        if url(pre_exec_file):
            try:
                contents = retrieve_remote_file(str(pre_exec_file))
            except RequestException as e:
                cli_logger.error(
                    f"Retrieving file from remote, '{pre_exec_file}', failed! Error:\n{e}"
                )
                raise typer.Exit(1)
        else:
            try:
                with open(pre_exec_file) as file:
                    contents = file.read()
            except FileNotFoundError:
                cli_logger.error(
                    "Failed to open pre-execution script. Couldn't find file: "
                    f"'{pre_exec_file}'!"
                )
                raise typer.Exit(1)
        self.config.pre_exec_command = contents

    def _set_exec_cmd(self):
        """Sets the execution command based on the given arguments. If the passed file is
        a url then it downloads its contents. If it's a file, then it tries to read it from
        the disk.
        """
        exec_file = self.config.exec_file
        if exec_file is None:
            return
        if self.config.exec_command is not None:
            return
        exec_file = str(exec_file)

        if url(exec_file):
            try:
                contents = retrieve_remote_file(str(exec_file))
            except RequestException as e:
                cli_logger.error(
                    f"Retrieving file from remote, '{exec_file}', failed! Error:\n{e}"
                )
                raise typer.Exit(1)
        else:
            try:
                with open(exec_file) as file:
                    contents = file.read()
            except FileNotFoundError:
                cli_logger.error(
                    f"Failed to open execution script. Couldn't find file: '{exec_file}'!"
                )
                raise typer.Exit(1)
        self.config.exec_command = contents

    def process_script(self) -> None:
        """Processes the given script by submitting and/or saving it, or by printing it to
        stdout. Also prepares the script by inserting default values wherever they are not
        set, as well as creating the needed directories.
        """
        self._set_default_config_fields()
        self._set_pre_exec_cmd()
        self._set_exec_cmd()
        script = self.config.generate_script()
        if not self.config.submit_job and not self.config.save_script:
            self.print_script(script=script)
            return

        if self.config.save_script:
            save_path = self.config.save_dir / f"{self.config.job_name}.slurm"
            save_path = save_path.resolve()
            self.save_script(script=script, file_path=save_path)

        if self.config.submit_job:
            self._ensure_log_dirs_exist()
            self.submit_script(script=script)

class MLSlurmBuilder(SlurmScriptBuilder):
    """Builds a SLURM script tailored to distributed machine learning.

    Uses the provided ``MLSlurmBuilderConfig`` to build the script and inserts values as
    needed.

    Args:
        config (MLSlurmBuilderConfig): Validated configuration controlling script generation.

    Note:
        The given configuration object might be modified by some of the methods.
    """

    config: MLSlurmBuilderConfig

    def __init__(self, config: MLSlurmBuilderConfig):
        super().__init__(config=config)
        self.config = config

    def _generate_job_identifier(self) -> str:
        num_nodes = self.config.num_nodes
        gpus_per_node = self.config.gpus_per_node
        identifier = (
            f"{'ray-' if self.config.use_ray else ''}"
            f"{self.config.distributed_strategy}-{num_nodes}x{gpus_per_node}"
        )
        return identifier

    def get_exec_command(self) -> str:
        """Generates an execution command for the SLURM script. Considers whether ray is
        enabled or not and finds the appropriate expected bash function.
        """
        if self.config.exec_command is not None:
            return self.config.exec_command

        if self.config.distributed_strategy == "horovod" and self.config.use_ray:
            cli_logger.error("Horovod together with Ray is not supported!")
            raise typer.Exit(1)
        if self.config.use_ray and self.config.py_spy:
            cli_logger.error("Ray together with py-spy profiling is not supported!")
            raise typer.Exit(1)
        if self.config.py_spy and self.config.distributed_strategy == "horovod":
            cli_logger.error("Horovod together with py-spy profiling is not supported!")
            raise typer.Exit(1)

        if self.config.use_ray:
            base_cmd = "ray-launcher"
        elif self.config.distributed_strategy == "horovod":
            base_cmd = "srun-launcher"
        elif self.config.distributed_strategy in ("ddp", "deepspeed"):
            base_cmd = "torchrun-launcher"
        else:
            cli_logger.error(f"Invalid strategy chosen: {self.config.distributed_strategy}")
            raise typer.Exit(1)

        if self.config.container_path:
            base_cmd += "-container"

        if self.config.py_spy and self.config.distributed_strategy in ("ddp", "deepspeed"):
            py_spy_profiling_filename = (
                rf"{self.config.distributed_strategy}_profile_\$SLURM_NODEID.txt"
            )
            py_spy_output_file = Path(DEFAULT_PY_SPY_DIR) / py_spy_profiling_filename
            base_cmd = (
                f"py-spy-{base_cmd} '{self.config.profiling_sampling_rate}' "
                f"'{py_spy_output_file}'"
            )

        base_cmd += f" '{self.config.build_training_command()}'"
        return base_cmd

    def get_pre_exec_command(self) -> str:
        """Generates a pre-execution command for the SLURM script. Adds a command to source
        the python venv if given and a command to export a container path variable if given.
        """
        pre_exec_command = ""
        if self.config.pre_exec_command is not None:
            pre_exec_command = self.config.pre_exec_command

        if self.config.python_venv is not None:
            pre_exec_command += f"\nsource {self.config.python_venv}/bin/activate"

        if self.config.container_path is not None:
            pre_exec_command += f"\nexport CONTAINER_PATH={self.config.container_path}"

        pre_exec_command = pre_exec_command.strip()
        return pre_exec_command

    def _set_default_config_fields(self) -> None:
        if self.config.job_name is None:
            self.config.job_name = self._generate_job_identifier()

        super()._set_default_config_fields()

    def process_script(self) -> None:
        """Generate the SLURM script then print, save, and/or submit based on config flags.

        - Always renders the script (filling defaults, loading exec/pre-exec files).
        - Prints to stdout when neither ``submit_job`` nor ``save_script`` is set.
        - Saves to ``save_dir`` when ``save_script`` is True.
        - Submits via ``sbatch`` when ``submit_job`` is True (ensures log dirs exist).
        """
        self._set_pre_exec_cmd()
        self._set_exec_cmd()
        self.config.pre_exec_command = self.get_pre_exec_command()
        self.config.exec_command = self.get_exec_command()
        self._set_default_config_fields()

        # Generate the script using the given configuration
        super().process_script()

    def process_all_strategies(
        self,
        strategies: Iterable[str] = ("ddp", "horovod", "deepspeed"),
    ):
        """Runs the SLURM script with all the given strategies."""
        original_config = self.config.model_copy(deep=True)
        original_run_name = self.config.run_name
        for strategy in strategies:
            if strategy == "horovod" and self.config.use_ray:
                continue
            updated_config = original_config.model_copy(deep=True)
            updated_config.distributed_strategy = strategy
            self.config = updated_config
            job_identifier = self._generate_job_identifier()

            self.config.run_name = f"{job_identifier}-{original_run_name}"

            # Updating job_name, std_out and err_out
            self.config.job_name = job_identifier
            std_out_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_identifier}.out"
            err_out_path = Path(DEFAULT_SLURM_LOG_DIR) / f"{job_identifier}.err"
            self.config.std_out = std_out_path
            self.config.err_out = err_out_path

            self.process_script()

    def process_scaling_test(
        self,
        strategies: Iterable[str] = ("ddp", "horovod", "deepspeed"),
        num_nodes_list: Iterable[int] = (1, 2, 4, 8),
    ):
        """Runs a scaling test, i.e. runs all the strategies with separate runs for each
        distinct number of nodes.
        """
        original_config = self.config.model_copy(deep=True)
        original_run_name = self.config.run_name
        for num_nodes in num_nodes_list:
            self.config = original_config.model_copy(deep=True)
            self.config.num_nodes = num_nodes
            total_gpus = self.config.num_nodes * self.config.gpus_per_node
            self.config.run_name = f"{original_run_name}-{total_gpus}gpu"
            self.process_all_strategies(strategies=strategies)


def generate_default_slurm_script(config: MLSlurmBuilderConfig) -> None:
    """Generates and optionally submits a default SLURM script from a validated config."""
    slurm_script_builder = MLSlurmBuilder(config)
    process_builder(slurm_script_builder=slurm_script_builder)


def process_builder(
    slurm_script_builder: MLSlurmBuilder,
):
    mode = slurm_script_builder.config.mode
    node_list = slurm_script_builder.config.scalability_nodes
    if mode == "single":
        slurm_script_builder.process_script()
    elif mode == "runall":
        slurm_script_builder.process_all_strategies()
    elif mode == "scaling-test":
        slurm_script_builder.process_scaling_test(num_nodes_list=node_list)
