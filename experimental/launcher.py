import datetime
import os
import shutil
import abc
import time
import uuid
from typing import Callable, Tuple, Any, Union, List, Optional

from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent
)
from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    DynamicRendezvousHandler
)
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
    C10dRendezvousBackend
)
from torch.distributed import TCPStore
from torch.distributed.elastic.multiprocessing import Std, start_processes

from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.distributed.run import config_from_args

from cluster import ClusterEnvironment, detect_cluster


class Launcher(abc.ABC):
    cluster: ClusterEnvironment

    @abc.abstractmethod
    def run(self, *args) -> Any:
        """Launches the distributed execution."""


class DummyTorchElasticLauncher(Launcher):
    """Simplified Torch Elastic launcher."""

    def __init__(
        self,
        cluster: Optional[ClusterEnvironment] = None,
        n_workers_per_node: int = 1,
        min_nodes: int = 1,
        max_nodes: int = 1,
        max_restarts: int = 1
    ) -> None:
        super().__init__()
        # detect_cluster() is preferred
        self.cluster = cluster if cluster is not None else detect_cluster()
        print(f"DummyTorchElasticLauncher with cluster '{self.cluster}'")
        self.n_workers_per_node = n_workers_per_node
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_restarts = max_restarts
        self.run_id = str(time.time())

        if cluster.creates_processes_externally and n_workers_per_node > 1:
            print("WARNING: the cluster may already spawn worker "
                  "processes for you... Consider setting "
                  "'n_workers_per_node=1'")

        g_world_size = cluster.num_nodes() * self.n_workers_per_node

        store = TCPStore(
            host_name=cluster.main_address,
            port=cluster.main_port,  # could conflict!
            world_size=g_world_size,
            is_master=cluster.global_rank() == 0,
            timeout=datetime.timedelta(seconds=3)
        )
        backend = C10dRendezvousBackend(store, self.run_id)
        self.rdzv_handler = DynamicRendezvousHandler.from_backend(
            run_id=self.run_id,
            store=store,
            backend=backend,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes
        )

    def run(
        self,
        func: Callable,
        args: Tuple = (),
        redirect: bool = False,
        log_dir: str = 'launcher_logs',
        tee_ranks: Union[str, int, List[int]] = None
    ) -> List[Any]:
        """Launches the distributed execution with Torch Elastic."""
        # Suppress all printing to console:
        # redirects={0: Std.ALL} # do no print, but save to file.
        # linked to Agent's log_dir
        redirects = Std.ALL if redirect else Std.NONE

        # Fore back printing to console, while redirecting to file
        # tee={0: Std.ALL} reactivates print to console + save to
        # log file for RANK 0
        if tee_ranks == 'all':
            tee = Std.ALL
        elif tee_ranks is None:
            tee = Std.NONE
        elif isinstance(tee_ranks, int):
            tee = {tee_ranks: Std.ALL}
        elif isinstance(tee_ranks, list):
            # tee_ranks is a list of int
            tee = {rnk: Std.ALL for rnk in tee_ranks}
        else:
            raise ValueError(f"unrecognized 'tee_ranks={tee_ranks}'")

        spec = WorkerSpec(
            role="worker",
            local_world_size=self.n_workers_per_node,
            entrypoint=func,
            args=args,
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.max_restarts,
            #   monitor_interval=monitor_interval,
            redirects=redirects,
            tee=tee
        )

        agent = LocalElasticAgent(spec, start_method="spawn", log_dir=log_dir)
        #   try:
        run_result = agent.run()
        if run_result.is_failed():
            print(f"worker 0 failed with: {run_result.failures[0]}")
            result = None
        else:
            print(f"worker 0 return value is: {run_result.return_values[0]}")
            result = run_result.return_values
        #   except Exception ex:
        #       # handle exception
        return result


class TorchElasticLauncher(Launcher):
    """
    Official Torch Elastic launcher.
    Does NOT support passing values as environment variables.

    Adapted from:
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py
    """

    def __init__(
        self,
        nnodes: str = '1:1',
        nproc_per_node: str = '1',
        rdzv_backend: str = 'static',
        rdzv_endpoint: str = '',
        rdzv_id: str = 'none',
        rdzv_conf: str = '',
        standalone: bool = False,
        max_restarts: int = 0,
        monitor_interval: float = 5,
        start_method: str = 'spawn',
        role: str = 'default',
        module: bool = False,
        no_python: bool = False,
        run_path: bool = False,
        log_dir: Optional[str] = None,
        redirects: str = '0',
        tee: str = '0',
        node_rank: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        local_addr: Optional[str] = None
    ) -> None:
        super().__init__()
        # emulate CLI args
        # TODO: include logic for 'action=check_env' or 'action=env'
        self.nnodes = nnodes
        self.nproc_per_node = nproc_per_node
        self.rdzv_backend = rdzv_backend
        self.rdzv_endpoint = rdzv_endpoint
        self.rdzv_id = rdzv_id
        self.rdzv_conf = rdzv_conf
        self.standalone = standalone
        self.max_restarts = max_restarts
        self.monitor_interval = monitor_interval
        self.start_method = start_method
        self.role = role
        self.module = module
        self.no_python = no_python
        self.run_path = run_path
        self.log_dir = log_dir
        self.redirects = redirects
        self.tee = tee
        self.node_rank = node_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.local_addr = local_addr
        # Placeholders
        self.training_script = "placeholder.py"
        self.training_script_args = []

    def config_from_args(
            self
    ) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
        return config_from_args(self)

    def run(
        self,
        func: Callable,
        args: Tuple = ()
    ) -> Any:
        if self.standalone:
            self.rdzv_backend = "c10d"
            self.rdzv_endpoint = "localhost:29400"
            self.rdzv_id = str(uuid.uuid4())
            # log.info(
            #     f"\n**************************************\n"
            #     f"Rendezvous info:\n"
            #     f"--rdzv_backend={self.rdzv_backend} "
            #     f"--rdzv_endpoint={self.rdzv_endpoint} "
            #     f"--rdzv_id={self.rdzv_id}\n"
            #     f"**************************************\n"
            # )

        config, _, _ = self.config_from_args()
        elastic_launch(
            config=config,
            entrypoint=func,
        )(*args)


class SimpleLauncher(Launcher):
    """Simple launcher based on multiprocessing.
    Use ONLY for single node applications.
    """

    def __init__(
        self,
        nproc_per_node: int,
        run_id: Optional[str] = None,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500
    ) -> None:
        super().__init__()
        self.nproc_per_node = nproc_per_node
        self.run_id = run_id if run_id is not None else f"RunID:{time.time()}"
        self.master_addr = master_addr
        self.master_port = master_port
        self.log_dir = f'{self.__class__.__name__}_logs'
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)

    def run(
        self,
        func: Callable,
        args: Tuple = ()
    ) -> Any:
        # Adapted from:
        # https://pytorch.org/docs/stable/elastic/multiprocessing.html
        w_args = {i: args for i in range(self.nproc_per_node)}
        # Emulates the env variables set by torch Elastic
        w_envs = {
            i: dict(
                RANK=str(i),
                LOCAL_RANK=str(i),
                GROUP_RANK=str(0),
                ROLE_RANK=str(i),
                WORLD_SIZE=str(self.nproc_per_node),
                LOCAL_WORLD_SIZE=str(self.nproc_per_node),
                ROLE_WORLD_SIZE=str(self.nproc_per_node),
                TORCHELASTIC_RUN_ID=str(self.run_id),
                MASTER_ADDR=str(self.master_addr),
                MASTER_PORT=str(self.master_port)
            )
            for i in range(self.nproc_per_node)
        }
        ctx = start_processes(
            name=self.__class__.__name__,
            entrypoint=func,
            args=w_args,
            envs=w_envs,
            log_dir=self.log_dir
        )
        ctx.wait()
        return ctx.return_values


class DeepSpeedLauncher(Launcher):
    """Official DeepSpeed launcher."""

    def __init__(self) -> None:
        super().__init__()

    def run(
        self,
        func: Callable,
        args: Tuple = ()
    ) -> Any:
        # TODO: complete
        raise NotImplementedError
