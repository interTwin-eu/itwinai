import datetime
import abc
import time
from typing import Callable, Tuple, Any, Union, List

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
from torch.distributed.elastic.multiprocessing import Std

# from lightning.pytorch.plugins.environments import (
#     ClusterEnvironment, SLURMEnvironment,
#     TorchElasticEnvironment, LightningEnvironment
# )

from cluster import ClusterEnvironment


class Launcher(abc.ABC):
    cluster: ClusterEnvironment

    @abc.abstractmethod
    def run(*args):
        """Launches the distributed execution."""


class DummyTorchElasticLauncher(Launcher):
    """Simplified Torch Elastic launcher."""

    def __init__(
        self,
        cluster: ClusterEnvironment,
        n_workers_per_node: int = 1,
        min_nodes: int = 1,
        max_nodes: int = 1,
        max_restarts: int = 1
    ) -> None:
        super().__init__()
        self.cluster = cluster
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
    ) -> Any:
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
    """Official Torch Elastic launcher."""


class SimpleLauncher(Launcher):
    """Simple launcher based on multiprocessing."""


class DeepSpeedLauncher(Launcher):
    """Official DeepSpeed launcher."""
