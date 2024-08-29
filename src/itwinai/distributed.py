import abc
import os
from pydantic import BaseModel
import builtins as __builtin__


class DistributedStrategy(abc.ABC):
    """Abstract class to define the distributed backend methods."""


class ClusterEnvironment(BaseModel):
    """Stores information about distributed environment."""
    #: Global rank of current worker, in a distributed environment.
    #: ``global_rank==0`` identifies the main worker.
    #: Defaults to 0.
    global_rank: int = 0
    #: Local rank of current worker, in a distributed environment.
    #: Defaults to 0.
    local_rank: int = 0
    #: Total number of workers in a distributed environment.
    #: Defaults to 1.
    global_world_size: int = 1
    #: Number of workers on the same node in a distributed environment.
    #: Defaults to 1.
    local_world_size: int = 1


def detect_distributed_environment() -> ClusterEnvironment:
    """Detects distributed environment, extracting information like
    global ans local ranks, and world size.
    """
    if os.getenv('TORCHELASTIC_RUN_ID') is not None:
        # Torch elastic environment
        # https://pytorch.org/docs/stable/elastic/run.html#environment-variables
        return ClusterEnvironment(
            global_rank=os.getenv('RANK'),
            local_rank=os.getenv('LOCAL_RANK'),
            local_world_size=os.getenv('LOCAL_WORLD_SIZE'),
            global_world_size=os.getenv('WORLD_SIZE')
        )
    elif os.getenv('OMPI_COMM_WORLD_SIZE') is not None:
        # Open MPI environment
        # https://docs.open-mpi.org/en/v5.0.x/tuning-apps/environment-var.html
        return ClusterEnvironment(
            global_rank=os.getenv('OMPI_COMM_WORLD_RANK'),
            local_rank=os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'),
            local_world_size=os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'),
            global_world_size=os.getenv('OMPI_COMM_WORLD_SIZE')
        )
    elif os.getenv('SLURM_JOB_ID') is not None:
        print("WARNING: detected SLURM environment, but "
              "unable to determine ranks and world sizes!")
        return ClusterEnvironment()
    else:
        return ClusterEnvironment()


#: Save original builtin print before patching it in distributed environments
builtin_print = __builtin__.print


def distributed_patch_print(is_main: bool) -> None:
    """Disable ``print()`` when not in main worker.

    Args:
        is_main (bool): whether it is called from main worker.
    """

    def print(*args, **kwself):
        """Print is disables on workers different from
        the main one, unless the print is called with
        ``force=True`` argument.
        """
        force = kwself.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwself)

    # Patch builtin print
    __builtin__.print = print
