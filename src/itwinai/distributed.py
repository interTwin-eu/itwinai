import abc
import os

from pydantic import BaseModel


class DistributedStrategy(abc.ABC):
    """Abstract class to define the distributed backend methods."""


class ClusterEnvironment(BaseModel):
    """Stores information about distributed environment."""
    global_rank: int = 0
    local_rank: int = 0
    local_world_size: int = 1
    global_world_size: int = 1


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
