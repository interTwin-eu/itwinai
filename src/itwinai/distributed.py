# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import builtins as __builtin__
import functools
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from ray.train import ScalingConfig


py_logger = logging.getLogger(__name__)


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


def ray_cluster_is_running() -> bool:
    try:
        # Run the `ray status` command. It should be less overhead than ray.init()
        result = subprocess.run(
            ["ray", "status"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Check if the output indicates a running cluster
        return (
            "Node status" in result.stdout
            and "Resources" in result.stdout
            and "Usage" in result.stdout
        )
    except subprocess.CalledProcessError:
        # If the command fails, the cluster is not running
        py_logger.debug(
            "Ray was checking for the existence of a Ray cluster by trying to "
            "connect to it, but could not do it. This is not a problem if you "
            "are not planning to connect to a Ray cluster."
        )
        return False
    except FileNotFoundError:
        # If `ray` command is not found, Ray is not installed
        py_logger.debug(
            "Error: 'ray' command not found while checking if a Ray cluster "
            "exists. Is Ray installed?"
        )
        return False


def _get_env(
    name: str,
    *,
    default: Any | None = None,
    cast: Callable[[str], Any] = lambda x: x,
    required: bool = False,
) -> Any:
    """Fetches and casts an environment variable.

    Args:
        name: the ENV var name.
        default: returned if var is unset (and required is False).
        cast: function to transform the raw str into the desired type.
        required: if True and var is missing, raises KeyError.
    """
    raw = os.getenv(name)
    if raw is None:
        if required:
            raise KeyError(f"Required environment variable {name!r} not set")
        return default
    try:
        return cast(raw)
    except Exception as e:
        py_logger.warning("Failed to cast env %s=%r using %s: %s", name, raw, cast, e)
        return default


def _get_int(name: str, default: int | None = None, required: bool = False) -> int:
    return _get_env(name, default=default, cast=int, required=required)


def detect_distributed_environment() -> ClusterEnvironment:
    """Detects a distributed environment by probing known env vars."""
    # 1) TorchElastic
    if os.getenv("TORCHELASTIC_RUN_ID") is not None:
        py_logger.debug("Using TorchElastic distributed cluster")
        # https://pytorch.org/docs/stable/elastic/run.html#environment-variables
        return ClusterEnvironment(
            global_rank=_get_int("RANK", required=True),
            local_rank=_get_int("LOCAL_RANK", required=True),
            local_world_size=_get_int("LOCAL_WORLD_SIZE", required=True),
            global_world_size=_get_int("WORLD_SIZE", required=True),
        )

    # 2) Open MPI (guard against stale SLURM_* that might be set)
    ompi_size = _get_int("OMPI_COMM_WORLD_SIZE", default=-1)
    slurm_tasks = _get_int("SLURM_NTASKS", default=0)
    if ompi_size >= slurm_tasks:
        py_logger.debug("Using Open MPI distributed cluster")
        # https://docs.open-mpi.org/en/v5.0.x/tuning-apps/environment-var.html
        return ClusterEnvironment(
            global_rank=_get_int("OMPI_COMM_WORLD_RANK", required=True),
            local_rank=_get_int("OMPI_COMM_WORLD_LOCAL_RANK", required=True),
            local_world_size=_get_int("OMPI_COMM_WORLD_LOCAL_SIZE", required=True),
            global_world_size=ompi_size,
        )

    # 3) SLURM (fallback)
    if os.getenv("SLURM_JOB_ID") is not None:
        py_logger.debug("Using SLURM distributed environment")
        # https://hpcc.umd.edu/hpcc/help/slurmenv.html
        return ClusterEnvironment(
            global_rank=_get_int("SLURM_PROCID", required=True),
            local_rank=_get_int("SLURM_LOCALID", required=True),
            local_world_size=_get_int("SLURM_NTASKS_PER_NODE", default=1),
            global_world_size=_get_int("SLURM_NTASKS", required=True),
        )

    # 4) default: no distributed env
    py_logger.debug("No distributed environment was detected")
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


def get_adaptive_ray_scaling_config() -> "ScalingConfig":
    """Returns a Ray scaling config for distributed ML training depending on the resources
    available in the Ray cluster. The number of workers is equal to the number of GPUs
    available, and if there are not GPUs two CPU-only workers are used.
    """
    import ray
    from ray.train import ScalingConfig

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Get cluster resources
    cluster_resources = ray.cluster_resources()
    num_gpus = int(cluster_resources.get("GPU", 0))

    # Configure ScalingConfig based on GPU availability
    if num_gpus <= 1:
        # If 0 or 1 GPU, don't use GPU for training
        py_logger.debug("Returning a scaling config to run distributed ML on 2 CPUs")
        return ScalingConfig(
            num_workers=2,  # Default to 2 CPU workers
            use_gpu=False,
        )
    else:
        # If multiple GPUs, use all available GPUs
        py_logger.debug(f"Returning a scaling config to run distributed ML on {num_gpus} GPUs")
        return ScalingConfig(num_workers=num_gpus, use_gpu=True)
