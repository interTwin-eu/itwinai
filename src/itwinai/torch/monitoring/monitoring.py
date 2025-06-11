# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import functools
import os
import time
from multiprocessing import Manager, Process
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import pandas as pd

from .backend import init_backend

if TYPE_CHECKING:
    from itwinai.torch.trainer import TorchTrainer

logging_columns = [
    "sample_idx",
    "utilization",
    "power",
    "local_rank",
    "node_idx",
    "num_global_gpus",
    "strategy",
    "probing_interval",
]


def probe_gpu_utilization_loop(
    node_idx: int,
    num_local_gpus: int,
    num_global_gpus: int,
    strategy_name: str,
    log_dict: Any,
    stop_flag: Any,
    probing_interval: int = 2,
    warmup_time: int = 5,
) -> None:
    """Logs the GPU utilization across all availble GPUs on a single node. Is meant to
    be called by multiprocessing's Process and expects variables to be shared using
    a multiprocessing.Manager object. Logs utilization into `log_dict` until
    stop_flag.value is set to True.

    Args:
        node_idx: The index of the compute node that the function is called by, used
            for logging purposes.
        num_local_gpus: Number of GPUs on the current compute node.
        num_global_gpus: Number of GPUs on all nodes combined.
        strategy: Which distributed strategy is being used, e.g. "ddp" or "horovod".
        log_dict: Dictionary for storing logging data on. Should be managed by a
            multiprocessing.Manager object.
        stop_flag: Shared value telling the function when to stop logging. Should be
            managed by a multiprocessing.Manager object.
        probing_interval: How long to wait between each time a read of the GPU
            utilization is done.
        warmup_time: How long to wait before logging starts, allowing the training to
            properly start before reading.

    """
    if not set(logging_columns).issubset(set(log_dict.keys())):
        missing_columns = set(logging_columns) - set(log_dict.keys())
        raise ValueError(f"log_dict is missing the following columns: {missing_columns}")

    # load management library backend
    man_lib_type, man_lib = init_backend()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if not cuda_visible_devices:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES environment variable is not set. "
            "Please set it to the indices of the GPUs you want to monitor."
        )

    gpu_ids = list(map(int, cuda_visible_devices.split(",")))

    # ensure all gpu handles where retrieved
    if len(gpu_ids) != num_local_gpus:
        raise ValueError(f"Expected {num_local_gpus} handles, but got {len(gpu_ids)}.")

    time.sleep(warmup_time)

    if man_lib_type == "nvidia":
        handles = [man_lib.nvmlDeviceGetHandleByIndex(idx) for idx in gpu_ids]
    elif man_lib_type == "amd":
        handles = amdsmi.amdsmi_get_processor_handles()
        # assumes that amdsmi_get_processor_handles() returns all GPUs on the node
        handles = [handles[id] for id in gpu_ids]  # filter handles by gpu_ids
    else:
        raise ValueError(f"Unsupported management library type: {man_lib_type}")

    sample_idx = 0
    while not stop_flag.value:
        for id, handle in zip(gpu_ids, handles):
            if man_lib_type == "nvidia":
                power = man_lib.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W

            elif man_lib_type == "amd":
                gpu_util = man_lib.amdsmi_get_gpu_activity(handle)["gfx_activity"]  # W
                power = man_lib.amdsmi_get_power_info(handle)["average_socket_power"]

            log_dict["sample_idx"].append(sample_idx)
            log_dict["utilization"].append(gpu_util)
            log_dict["power"].append(power)
            log_dict["local_rank"].append(id)
            log_dict["node_idx"].append(node_idx)
            log_dict["num_global_gpus"].append(num_global_gpus)
            log_dict["strategy"].append(strategy_name)
            log_dict["probing_interval"].append(probing_interval)

        sample_idx += 1
        time.sleep(probing_interval)


def measure_gpu_utilization(method: Callable) -> Callable:
    """Decorator for measuring GPU utilization and storing it to a .csv file."""

    def write_logs_to_file(utilization_logs: List[Dict], output_path: Path) -> None:
        dataframes = []
        for log in utilization_logs:
            if len(log) == 0:
                continue
            dataframes.append(pd.DataFrame(log))

        log_df = pd.concat(dataframes)
        log_df.to_csv(output_path, index=False)
        print(f"Writing GPU energy dataframe to '{output_path.resolve()}'.")

    @functools.wraps(method)
    def measured_method(self: "TorchTrainer", *args, **kwargs) -> Any:
        if not self.measure_gpu_data:
            print("Warning: Profiling of GPU data has been disabled!")
            return method(self, *args, **kwargs)

        gpu_probing_interval = 1
        warmup_time = 5

        strategy = self.strategy
        strategy_name = strategy.name

        local_rank = strategy.local_rank()
        global_rank = strategy.global_rank()
        num_global_gpus = strategy.global_world_size()
        num_local_gpus = strategy.local_world_size()
        node_idx = global_rank // num_local_gpus

        gpu_monitor_process = None
        manager = None
        stop_flag = None
        data = None

        # Starting a child process once per node
        if local_rank == 0:
            # Setting up shared variables for the child process
            manager = Manager()
            data = manager.dict()
            for col in logging_columns:
                data[col] = manager.list()
            stop_flag = manager.Value("i", False)

            gpu_monitor_process = Process(
                target=probe_gpu_utilization_loop,
                kwargs={
                    "node_idx": node_idx,
                    "num_local_gpus": num_local_gpus,
                    "num_global_gpus": num_global_gpus,
                    "strategy_name": strategy_name,
                    "log_dict": data,
                    "stop_flag": stop_flag,
                    "probing_interval": gpu_probing_interval,
                    "warmup_time": warmup_time,
                },
            )
            gpu_monitor_process.start()

        local_utilization_log = {}
        try:
            result = method(self, *args, **kwargs)
        finally:
            if local_rank == 0:
                stop_flag.value = True
                grace_period = 5  # extra time to let process finish gracefully
                gpu_monitor_process.join(timeout=gpu_probing_interval + grace_period)

                # Converting the shared log to non-shared log
                local_utilization_log = {key: list(data[key]) for key in data.keys()}
                manager.shutdown()

        global_utilization_log = strategy.gather_obj(local_utilization_log, dst_rank=0)
        if strategy.is_main_worker:
            output_dir = Path(f"scalability-metrics/{self.run_id}/gpu-energy-data")
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f"{strategy_name}_{num_global_gpus}.csv"

            write_logs_to_file(global_utilization_log, output_path)

        return result

    return measured_method
