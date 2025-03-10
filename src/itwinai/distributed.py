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
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from ray.train import ScalingConfig


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
        return False
    except FileNotFoundError:
        # If `ray` command is not found, Ray is not installed
        return False


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
    # It is difficult to understand ranks from a Ray cluster... It could have been set up to
    # tune a model with non-distributed strategy.
    # elif ray_cluster_is_running():
    #     import ray

    #     ray_initialized_in_here = False

    #     if not ray.is_initialized():
    #         ray_initialized_in_here = True
    #         ray.init(address="auto")
    #     try:
    #         # Determine the local rank and local world size
    #         current_node = ray.util.get_node_ip_address()
    #         all_nodes = [node["NodeManagerAddress"] for node in ray.nodes()]

    #         # Filter tasks on the same node
    #         local_world_size = all_nodes.count(current_node)
    #         local_rank = (
    # all_nodes[: all_nodes.index(current_node) + 1].count(current_node) - 1
    # )
    #         cluster = ClusterEnvironment(
    #             global_rank=ray.get_runtime_context().get_node_id(),
    #             local_rank=local_rank,
    #             local_world_size=local_world_size,
    #             global_world_size=len(ray.nodes()),
    #         )
    #     finally:
    #         if ray_initialized_in_here:
    #             ray.shutdown()
    #     return cluster
    elif os.getenv("SLURM_JOB_ID") is not None:
        # https://hpcc.umd.edu/hpcc/help/slurmenv.html
        return ClusterEnvironment(
            global_rank=os.getenv("SLURM_PROCID"),
            local_rank=os.getenv("SLURM_LOCALID"),
            local_world_size=os.getenv("SLURM_NTASKS_PER_NODE", 1),
            global_world_size=os.getenv("SLURM_NTASKS"),
        )
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


def get_adaptive_ray_scaling_config() -> "ScalingConfig":
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
        return ScalingConfig(
            num_workers=2,  # Default to 2 CPU workers
            use_gpu=False,
        )
    else:
        # If multiple GPUs, use all available GPUs
        return ScalingConfig(num_workers=num_gpus, use_gpu=True)
