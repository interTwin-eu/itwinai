from pathlib import Path

from pydantic import BaseModel

from itwinai.slurm.slurm_constants import SLURM_TEMPLATE


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
    memory: str
    exclusive: bool = False

    # Typically used to set up the environment before executing the command,
    # e.g. "ml Python", "source .venv/bin/activate" etc.
    pre_exec_command: str | None = None

    # Command to execute, typically an 'srun' command
    exec_command: str | None = None

    def exclusive_line(self) -> str:
        return "#SBATCH --exclusive" if self.exclusive else ""

    def generate_script(self) -> str:
        """Uses the provided configuration parameters and formats a SLURM script with
        the requested settings.

        Returns:
            str: A string containing the SLURM script.
        """
        if (
            self.std_out is None
            or self.err_out is None
            or self.job_name is None
            or self.pre_exec_command is None
            or self.exec_command is None
        ):
            raise ValueError(
                "SlurmScriptConfiguration has some fields set to None! Make sure to set all"
                " fields before generating script! Configuration was formatted as follows:\n"
                f"{repr(self)}"
            )

        return SLURM_TEMPLATE.format_map(
            self.model_dump() | {"exclusive_line": self.exclusive_line()}
        )
