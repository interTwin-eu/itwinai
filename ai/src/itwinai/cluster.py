"""Cluster environments where to run AI workflows. Partially adapted from:
https://github.com/facebookresearch/detr/blob/master/util/misc.py and
https://github.com/ramyamounir/Template/blob/main/lib/utils/distributed.py
"""

from typing import Optional
from abc import ABCMeta, abstractmethod
import os
import signal
import subprocess
from pathlib import Path
from contextlib import contextmanager

import submitit
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from .backend.torch.types import TorchDistributedBackend as BackendT


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*self, **kwself):
        force = kwself.pop('force', False)
        if is_master or force:
            builtin_print(*self, **kwself)

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
    world_size: int = -1
    rank: int = -1
    rnd_seed: int = None
    distributed: bool = False
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

    def is_main(self) -> bool:
        return self.rank == 0

    def cleanup_resources(self):
        dist.barrier()
        dist.destroy_process_group()

    @abstractmethod
    @contextmanager
    def init_dist_gpu(self, *args, **kwargs):
        pass


class LocalCluster(ClusterEnvironment):
    """Simple single node cluster with access to multiple GPUs."""

    def __init__(
        self,
        backend: Optional[str] = None,
        gpus: Optional[str] = None,
        # max_gpus: int = -1,
        port: int = 49153,
        rnd_seed: Optional[int] = 42
    ) -> None:
        """Initialize local cluster for multi-GPU access.

        Args:
            backend (Optional[str], optional): supported PyTorch backends.
                If None, workload is not distributed. Defaults to None.
            gpus (Optional[str], optional): list of visible GPU devices
                (e.g., '1,2,3'). If None, CPU is used. Defaults to None.
            port (int, optional): TCP port used by the master process.
                Defaults to 49153.
            rnd_seed (Optional[int], optional): random seed to be setup after
                all processes are setup. Defaults to 42.
        """

        super().__init__()
        self.backend = backend
        self.gpus = gpus
        self.port = port
        self.rnd_seed = rnd_seed
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus

        self.ngpus_per_node = torch.cuda.device_count()
        self.rank = 0
        self.dist_url = f'tcp://localhost:{self.port}'
        self.world_size = self.ngpus_per_node

        self.distributed = True
        self.use_cuda = True
        if self.backend is None or self.ngpus_per_node <= 1:
            print("Distributed has been disabled.")
            self.distributed = False
            self.dist_url = None
            self.world_size = 1
            self.rank = 0
        if self.gpus is None or not torch.cuda.is_available():
            print("Cuda disabled... Running on single CPU.")
            self.use_cuda = False
            self.distributed = False
            self.dist_url = None
            self.world_size = 1
            self.rank = 0

    @contextmanager
    def init_dist_gpu(self, worker_id) -> torch.device:
        if self.distributed:
            torch.cuda.set_device(worker_id)
            self.gpu = worker_id
            self.rank += self.gpu

            try:
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.dist_url,
                    world_size=self.world_size,
                    rank=self.rank
                )
                fix_random_seeds(self.rnd_seed)
                torch.cuda.set_device(self.gpu)
                cudnn.benchmark = True
                dist.barrier()

                setup_for_distributed(self.is_main())
                print("SETUP DISTRIBUTED COMPLETE")
                yield torch.device('cuda', worker_id)
            finally:
                self.cleanup_resources()
        else:
            # Distributed is disabled
            if self.use_cuda:
                torch.cuda.set_device(worker_id)
                yield torch.device('cuda', worker_id)
            else:
                yield torch.device('cpu')


class SLURMCluster(ClusterEnvironment):
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
        self.rank = int(os.getenv('SLURM_NODEID')) * self.ngpus_per_node
        self.world_size = int(os.getenv('SLURM_NNODES')) * self.ngpus_per_node

    @contextmanager
    def init_dist_gpu(self):
        try:
            job_env = submitit.JobEnvironment()
            self.output_dir = Path(
                str(self.output_dir).replace("%j", str(job_env.job_id)))
            self.gpu = job_env.local_rank
            self.rank = job_env.global_rank

            dist.init_process_group(
                backend=self.backend,
                init_method=self.dist_url,
                world_size=self.world_size,
                rank=self.rank
            )
            fix_random_seeds(self.rnd_seed)
            torch.cuda.set_device(self.gpu)
            cudnn.benchmark = True
            dist.barrier()

            setup_for_distributed(self.is_main())
            yield
        finally:
            self.cleanup_resources()
