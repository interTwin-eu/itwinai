import os
import datetime
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import DynamicRendezvousHandler
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import C10dRendezvousBackend
from torch.distributed import TCPStore
from torch.distributed.elastic.multiprocessing import Std

nproc_per_node = 4
max_restarts = 2


def trainer_entrypoint_fn(a):
    print(f"{a}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} {os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")
    return 123


if __name__ == "__main__":
    store = TCPStore(host_name="localhost", port=29400,
                     world_size=nproc_per_node, is_master=True, timeout=datetime.timedelta(seconds=3))
    backend = C10dRendezvousBackend(store, "my_run_id")
    rdzv_handler = DynamicRendezvousHandler.from_backend(
        run_id="my_run_id",
        store=store,
        backend=backend,
        min_nodes=1,
        max_nodes=1
    )
    spec = WorkerSpec(
        role="trainer",
        local_world_size=nproc_per_node,
        entrypoint=trainer_entrypoint_fn,
        args=("foobar",),
        rdzv_handler=rdzv_handler,
        max_restarts=max_restarts,
        #   monitor_interval=args.monitor_interval,
        # # redirects={0: Std.ALL} # do no print, but save to file. linked to Agent's log_dir
        redirects=Std.ALL,  # suppress all printing to console
        # tee={0: Std.ALL} reactivates print to console + save to log file for RANK 0
        tee={0: Std.ALL}
    )

    agent = LocalElasticAgent(spec, start_method="spawn", log_dir='logs')
    #   try:
    run_result = agent.run()
    if run_result.is_failed():
        print(f"worker 0 failed with: {run_result.failures[0]}")
    else:
        print(f"worker 0 return value is: {run_result.return_values[0]}")
    #   except Exception ex:
    #       # handle exception
