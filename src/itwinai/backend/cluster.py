"""Cluster environments where to run AI workflows. Partially adapted from:
https://github.com/facebookresearch/detr/blob/master/util/misc.py and
https://github.com/ramyamounir/Template/blob/main/lib/utils/distributed.py
"""

from __future__ import annotations
from typing import Optional
from abc import ABCMeta, abstractmethod
import os
import signal
import subprocess
from pathlib import Path
from contextlib import contextmanager

import numpy as np


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


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
        if backend_name not in BackendT:
            raise ValueError(
                "Unrecognized 'backend' field. Allowed values "
                f"are: {BackendT.list()}. Received '{backend_name}'")
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


class TorchCluster(ClusterEnvironment):
    def __init__(self) -> None:
        import torch
        import torch.distributed as dist
        import torch.backends.cudnn as cudnn
        from .torch.types import TorchDistributedBackend as BackendT
        super().__init__()

    def is_cuda_available(self) -> bool:
        return self.use_cuda and torch.cuda.is_available()

    def is_main_worker(self) -> bool:
        """Checks if the current process is the main/master process
        in the whole job.
        """
        return self.global_rank == 0

    def cleanup_resources(self):
        dist.barrier()
        dist.destroy_process_group()


class LocalCluster(TorchCluster):
    """Simple single node cluster with optional access to multiple GPUs."""

    def __init__(
        self,
        backend: Optional[str] = None,
        gpus: Optional[str] = '',
        port: int = 49153,
        rnd_seed: Optional[int] = 42
    ) -> None:
        """Initialize local cluster for multi-GPU access.

        Args:
            backend (Optional[str], optional): supported PyTorch backends.
                If None, workload is not distributed. Defaults to None.
            gpus (Optional[str], optional): list of visible GPU devices
                (e.g., '1,2,3'). If empty string uses all available GPUs.
                If None, CPU is used. Defaults to ''.
            port (int, optional): TCP port used by the master process.
                Defaults to 49153.
            rnd_seed (Optional[int], optional): random seed to be setup after
                all processes are setup. Defaults to 42.
        """
        super().__init__()
        self.backend = backend
        self.gpus = gpus
        self.port = port
        self.dist_url = f'tcp://127.0.0.1:{self.port}'
        self.rnd_seed = rnd_seed

        if self.gpus != '' and self.gpus is not None:
            # Restrict the number of GPUs visible according to user needs
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus

        self.ngpus_per_node = torch.cuda.device_count()
        self.global_rank = 0
        self.global_world_size = self.ngpus_per_node

        print(f"{self.ngpus_per_node} GPUs are available.")
        self.distributed = True
        # This flag tells whether the user wants to use the GPU(s)
        self.use_cuda = (
            self.gpus is not None  # GPU is not manually disabled
            and torch.cuda.device_count() >= 1  # At least one GPU is selected
        )
        if self.backend is None or self.ngpus_per_node <= 1:
            print("Distributed has been disabled.")
            self.distributed = False
            self.dist_url = None
            self.global_world_size = 1
            self.global_rank = 0
        if not self.is_cuda_available():
            print("CUDA disabled... Running on single CPU.")
            self.use_cuda = False
            self.distributed = False
            self.dist_url = None
            self.global_world_size = 1
            self.global_rank = 0

        # Since single node case
        self.local_world_size = self.global_world_size

    @contextmanager
    def init_dist_gpu(self, worker_id) -> torch.device:
        if self.distributed:
            torch.cuda.set_device(worker_id)
            self.global_rank += worker_id
            # print(f'GLOBAL RANK: {self.global_rank}')
            # Since single node case
            self.local_rank = self.global_rank
            # Simplification: worker ID mapped to GPU ID
            self.gpu_id = worker_id

            try:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.dist_url,
                    world_size=self.global_world_size,
                    rank=self.global_rank
                )
                fix_random_seeds(self.rnd_seed)
                torch.cuda.set_device(self.gpu_id)
                cudnn.benchmark = True
                dist.barrier()

                setup_for_distributed(self.is_main_worker())
                print("SETUP DISTRIBUTED COMPLETE")
                yield torch.device('cuda', worker_id)
            finally:
                self.cleanup_resources()
        else:
            # Distributed is disabled
            # Since single node case
            self.global_rank = 0
            self.local_rank = self.global_rank
            if self.use_cuda:
                torch.cuda.set_device(worker_id)
                yield torch.device('cuda', worker_id)
            else:
                yield torch.device('cpu')


class SLURMCluster(TorchCluster):
    """SLURM cluster with access to multi-node multi-GPU."""

    def __init__(
            self,
            port: int = 49153,
            backend: str = 'gloo',
            rnd_seed: Optional[int] = 42
    ) -> None:
        super().__init__()
        self.port = port
        self.backend = backend
        self.rnd_seed = rnd_seed
        if 'SLURM_JOB_ID' not in os.environ:
            raise RuntimeError(
                "'SLURM_JOB_ID' environment variable is not set. "
                "Perhaps you are not running in a slurm cluster?"
            )

        self.ngpus_per_node = torch.cuda.device_count()

        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # find a common host name on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        self.dist_url = f'tcp://{host_name}:{self.port}'

        # distributed parameters
        self.global_rank = int(os.getenv('SLURM_NODEID')) * self.ngpus_per_node
        self.global_world_size = int(
            os.getenv('SLURM_NNODES')) * self.ngpus_per_node

    @contextmanager
    def init_dist_gpu(self):
        import submitit
        try:
            job_env = submitit.JobEnvironment()
            self.output_dir = Path(
                str(self.output_dir).replace("%j", str(job_env.job_id)))
            self.gpu = job_env.local_rank
            self.global_rank = job_env.global_rank

            dist.init_process_group(
                backend=self.backend,
                init_method=self.dist_url,
                world_size=self.global_world_size,
                rank=self.global_rank
            )
            fix_random_seeds(self.rnd_seed)
            torch.cuda.set_device(self.gpu)
            cudnn.benchmark = True
            dist.barrier()

            setup_for_distributed(self.is_main_worker())
            yield
        finally:
            self.cleanup_resources()
