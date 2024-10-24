import functools
import sys
import time
from multiprocessing import Manager, Process
from typing import Callable, Any, List, Dict
from pathlib import Path

import pandas as pd
import pynvml
import torch
from itwinai.torch.distributed import TorchDDPStrategy, NonDistributedStrategy, HorovodStrategy, DeepSpeedStrategy
from net import Net
from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit
from torchvision import datasets, transforms

from itwinai.loggers import ConsoleLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer

logging_columns = [
    "sample_idx",
    "utilization",
    "power",
    "local_rank",
    "node_idx",
    "global_num_gpus",
    "strategy",
]


def log_gpu_utilization(
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
        local_num_gpus: Number of GPUs on the current compute node.
        global_num_gpus: Number of GPUs on all nodes combined.
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
        raise ValueError(
            f"log_dict is missing the following columns: {missing_columns}"
        )

    nvmlInit()
    time.sleep(warmup_time)

    sample_idx = 0
    while not stop_flag.value:
        for idx in range(num_local_gpus):
            handle = nvmlDeviceGetHandleByIndex(idx)
            utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)

            gpu_util = utilization_rates.gpu
            power = pynvml.nvmlDeviceGetPowerUsage(handle)  
            power = power / 1000 # mW -> W

            log_dict["sample_idx"].append(sample_idx)
            log_dict["utilization"].append(gpu_util)
            log_dict["power"].append(power)
            log_dict["local_rank"].append(idx)
            log_dict["node_idx"].append(node_idx)
            log_dict["global_num_gpus"].append(num_global_gpus)
            log_dict["strategy"].append(strategy_name)

        sample_idx += 1

        time.sleep(probing_interval)


def measure_gpu_utilization(method: Callable) -> Callable:
    """Decorator for measuring GPU utilization and storing it to a .csv file."""

    def write_logs_to_csv(utilization_logs: List[Dict], output_path: Path) -> None:
        dataframes = []
        for log in utilization_logs:
            if len(log) == 0:
                continue
            dataframes.append(pd.DataFrame(log))
        log_df = pd.concat(dataframes)
        log_df.to_csv(output_path, index=False)
        print(f"Writing DataFrame to '{output_path}'.")

    @functools.wraps(method)
    def measured_method(self: TorchTrainer, *args, **kwargs) -> Any:
        gpu_probing_interval = 1
        warmup_time = 5

        strategy = self.strategy
        strategy.init()

        if isinstance(strategy, NonDistributedStrategy):
            strat_name = "non-dist"
        elif isinstance(strategy, TorchDDPStrategy):
            strat_name = "ddp"
        elif isinstance(strategy, DeepSpeedStrategy):
            strat_name = "deepspeed"
        elif isinstance(strategy, HorovodStrategy):
            strat_name = "horovod"
        else:
            strat_name = "unk"

        local_rank = strategy.local_rank()
        global_rank = strategy.global_rank()
        num_global_gpus = strategy.global_world_size()
        num_local_gpus = torch.cuda.device_count()
        node_idx = global_rank // num_local_gpus

        output_path = Path(f"utilization_logs/dataframe_{strat_name}_{num_global_gpus}.csv")
        output_path.parent.mkdir(exist_ok=True, parents=True)

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
                target=log_gpu_utilization,
                kwargs={
                    "node_idx": node_idx,
                    "num_local_gpus": num_local_gpus,
                    "num_global_gpus": num_global_gpus,
                    "strategy_name": strat_name,
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
            write_logs_to_csv(global_utilization_log, output_path)

        strategy.clean_up()
        return result

    return measured_method


def train(args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    root_dir = Path("data")
    train_dataset = datasets.MNIST(
        str(root_dir), train=True, download=True, transform=transform
    )
    validation_dataset = datasets.MNIST(str(root_dir), train=False, transform=transform)

    # Neural network to train
    model = Net()

    training_config = TrainingConfiguration(
        batch_size=args.batch_size,
        optim_lr=args.lr,
        optimizer="adadelta",
        loss="cross_entropy",
        num_workers_dataloader=1,
    )
    logger = ConsoleLogger()

    trainer = TorchTrainer(
        config=training_config,
        model=model,
        strategy=args.strategy,
        epochs=args.epochs,
        logger=logger,
    )
    trainer.execute(train_dataset, validation_dataset)


class UtilizationTrainer(TorchTrainer):

    @measure_gpu_utilization
    def execute(self, train_dataset, validation_dataset=None, test_dataset=None):
        return super().execute(train_dataset, validation_dataset, test_dataset)


def main():
    strategy = "horovod"
    epochs = 10
    lr = 1e-4
    batch_size = 64

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    root_dir = Path("data")
    train_dataset = datasets.MNIST(
        str(root_dir),
        train=True,
        download=True,
        transform=transform,
    )
    validation_dataset = datasets.MNIST(str(root_dir), train=False, transform=transform)

    # Neural network to train
    model = Net()

    training_config = TrainingConfiguration(
        batch_size=batch_size,
        optim_lr=lr,
        optimizer="adadelta",
        loss="cross_entropy",
        num_workers_dataloader=1,
    )
    logger = ConsoleLogger()

    trainer = UtilizationTrainer(
        config=training_config,
        model=model,
        strategy=strategy,
        epochs=epochs,
        logger=logger,
    )

    trainer.execute(train_dataset, validation_dataset)


if __name__ == "__main__":
    main()
    sys.exit()
