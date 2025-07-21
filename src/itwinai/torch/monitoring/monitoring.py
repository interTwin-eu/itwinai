# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import functools
import logging
import time
from multiprocessing import Manager, Process
from multiprocessing.managers import ValueProxy
from typing import TYPE_CHECKING, Any, Callable

from ...loggers import Logger, LoggersCollection
from .backend import detect_gpu_backend

if TYPE_CHECKING:
    from itwinai.torch.trainer import TorchTrainer

py_logger = logging.getLogger(__name__)

cli_logger = logging.getLogger("cli_logger")


def profile_gpu_utilization(
    stop_flag: ValueProxy,
    local_rank: int,
    global_rank: int,
    global_world_size: int,
    strategy_name: str,
    logger: Logger,
    run_id: str | None = None,
    parent_run_id: str | None = None,
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
    backend = detect_gpu_backend()
    visible_gpu_ids = backend.get_visible_gpu_ids()

    if not visible_gpu_ids:
        py_logger.warning("No visible GPUs found. Skipping GPU utilization profiling.")
        return

    if local_rank >= len(visible_gpu_ids):
        raise ValueError("local_rank exceeds the number of visible GPUs.")

    gpu_handle = backend.get_handle_by_id(visible_gpu_ids[local_rank])

    # warmup time to wait for the training to start
    time.sleep(warmup_time)

    sample_idx = 0

    # Make sure logger context can be created again for this process
    if isinstance(logger, LoggersCollection):
        for sublogger in logger.loggers:
            sublogger.is_initialized = False

    logger.is_initialized = False

    # attach to the worker run
    logger.create_logger_context(
        rank=global_rank,
        parent_run_id=parent_run_id,
        run_id=run_id,
    )
    logger.log(
        item=strategy_name,
        identifier="strategy",
        kind="param",
    )
    logger.log(
        item=global_rank,
        identifier="global_rank",
        kind="param",
    )
    logger.log(
        item=global_world_size,
        identifier="num_global_gpus",
        kind="param",
    )
    logger.log(
        item=probing_interval,
        identifier="probing_interval",
        kind="param",
    )

    t_start = time.monotonic()  # fractional seconds

    while not stop_flag.value:
        time_stamp = time.monotonic() - t_start

        gpu_util = backend.get_gpu_utilization(gpu_handle)
        gpu_power = backend.get_gpu_power_usage(gpu_handle)

        logger.log(
            item=gpu_power,
            identifier="gpu_power_W",
            kind="metric",
            step=int(time_stamp),
        )
        logger.log(
            item=gpu_util,
            identifier="gpu_utilization_percent",
            kind="metric",
            step=int(time_stamp),
        )

        sample_idx += 1
        time.sleep(probing_interval)

    logger.destroy_logger_context()


def measure_gpu_utilization(method: Callable) -> Callable:
    """Decorator for measuring GPU utilization and storing it to a .csv file."""

    @functools.wraps(method)
    def measured_method(self: "TorchTrainer", *args, **kwargs) -> Any:
        if not self.measure_gpu_data:
            cli_logger.warning("Profiling of GPU data has been disabled!")
            return method(self, *args, **kwargs)

        if not self.logger:
            py_logger.warning(
                f"No loggers set, while measure_gpu_data is set to {self.measure_gpu_data}"
                " Please provide loggers so measure_gpu_data can log."
                " Skipping GPU logging."
            )
            return method(self, *args, **kwargs)

        gpu_probing_interval = 1
        warmup_time = 5

        strategy = self.strategy

        run_id = self.mlflow_worker_run_id
        parent_run_id = self.mlflow_train_run_id

        local_rank = strategy.local_rank()
        global_rank = strategy.global_rank()

        manager = Manager()
        stop_flag = manager.Value("i", False)

        gpu_monitor_process = Process(
            target=profile_gpu_utilization,
            kwargs={
                "stop_flag": stop_flag,
                "local_rank": local_rank,
                "global_rank": global_rank,
                "global_world_size": strategy.global_world_size(),
                "strategy_name": strategy.name,
                "logger": self.logger,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "probing_interval": gpu_probing_interval,
                "warmup_time": warmup_time,
            },
        )
        # Set child process as daemon such that child exits when parent exits
        gpu_monitor_process.daemon = True
        gpu_monitor_process.start()

        try:
            result = method(self, *args, **kwargs)
        finally:
            # Terminate the process
            stop_flag.value = True
            grace_period = 4  # seconds
            timeout = gpu_probing_interval + grace_period
            gpu_monitor_process.join(timeout=timeout)

        return result

    return measured_method
