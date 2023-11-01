"""Cluster environments where to run AI workflows."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
import os
from contextlib import contextmanager


def setup_for_distributed(is_main):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwself):
        force = kwself.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwself)

    __builtin__.print = print


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class ClusterEnvironment(metaclass=ABCMeta):
    port: int = -1
    ngpus_per_node: int = -1
    global_world_size: int = -1
    global_rank: int = -1
    local_world_size: int = -1
    local_rank: int = -1
    rnd_seed: int = None
    distributed: bool = False
    # This flag tells whether the user wants to use the GPU(s)
    use_cuda: bool = False

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, backend_name: str) -> None:
        self._set_backend(backend_name)

    def _set_backend(self, backend_name: str) -> None:
        # Override to implement sanitization
        self._backend = backend_name

    @abstractmethod
    def is_main_worker(self) -> bool:
        """Tells if the current process is the main/master process."""
        pass

    @abstractmethod
    def is_cuda_available(self) -> bool:
        pass

    @abstractmethod
    @contextmanager
    def init_dist_gpu(self, *args, **kwargs):
        pass

    def cleanup_resources(self):
        pass
    