# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import abc
import builtins as __builtin__
import functools
import os
import sys
from typing import Any, Callable

from pydantic import BaseModel


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
    if os.getenv("TORCHELASTIC_RUN_ID") is not None:
        # Torch elastic environment
        # https://pytorch.org/docs/stable/elastic/run.html#environment-variables
        return ClusterEnvironment(
            global_rank=os.getenv("RANK"),
            local_rank=os.getenv("LOCAL_RANK"),
            local_world_size=os.getenv("LOCAL_WORLD_SIZE"),
            global_world_size=os.getenv("WORLD_SIZE"),
        )
    elif os.getenv("OMPI_COMM_WORLD_SIZE") is not None:
        # Open MPI environment
        # https://docs.open-mpi.org/en/v5.0.x/tuning-apps/environment-var.html
        return ClusterEnvironment(
            global_rank=os.getenv("OMPI_COMM_WORLD_RANK"),
            local_rank=os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"),
            local_world_size=os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"),
            global_world_size=os.getenv("OMPI_COMM_WORLD_SIZE"),
        )
    elif os.getenv("SLURM_JOB_ID") is not None:
        print(
            "WARNING: detected SLURM environment, but "
            "unable to determine ranks and world sizes!"
        )
        return ClusterEnvironment()
    else:
        return ClusterEnvironment()


#: Save original builtin print before patching it in distributed environments
builtin_print = __builtin__.print


def distributed_patch_print(is_main: bool) -> Callable:
    """Disable ``print()`` when not in main worker.

    Args:
        is_main (bool): whether it is called from main worker.

    Returns:
        Callable: patched ``print()``.
    """

    def patched_print(*args, **kwself):
        """Print is disables on workers different from
        the main one, unless the print is called with
        ``force=True`` argument.
        """
        force = kwself.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwself)

    return patched_print


def suppress_workers_print(func: Callable) -> Callable:
    """Decorator to suppress ``print()`` calls in workers having global rank
    different from 0. To force printing on all workers you need to use
    ``print(..., force=True)``.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Disable print in workers different from the main one,
        # when in distributed environments.
        dist_grank = detect_distributed_environment().global_rank
        patched_print = distributed_patch_print(is_main=dist_grank == 0)
        previous_print_backup = __builtin__.print
        __builtin__.print = patched_print
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            # Reset print to builtin before raising the exception.
            __builtin__.print = previous_print_backup
            raise exc
        # Reset print to builtin
        __builtin__.print = previous_print_backup
        return result

    return wrapper


def suppress_workers_output(func):
    """Decorator to suppress ``stadout`` and ``stderr`` in workers having global rank
    different from 0.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Get global rank
        dist_grank = detect_distributed_environment().global_rank
        try:
            if dist_grank == 0:
                # If on main worker
                return func(*args, **kwargs)

            # If not on main worker, redirect stdout and stderr to devnull
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                # Execute the wrapped function
                return func(*args, **kwargs)
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return wrapper
