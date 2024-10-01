"""
Adapted from: https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

import argparse
import time
from pathlib import Path
from typing import Dict
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision import datasets, transforms

from itwinai.loggers import ConsoleLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.distributed import TorchDistributedStrategy

import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    """Class for monitoring GPU utilization using a different thread than the main one. 
    Works by sleeping for a certain amount of time, specified by `delay`, allowing 
    the GIL in Python to run other threads in the mean time. 
    """
    def __init__(self, delay: int, strategy: TorchDistributedStrategy, global_rank: int) -> None:
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay 
        self.strategy: TorchDistributedStrategy = strategy
        self.start_time = time.time()
        self.monitoring_log: Dict = {}
        self.global_rank = global_rank
        self.start()


    def run(self) -> None:
        if not self.strategy.global_rank() == self.global_rank: 
            return

        local_gpus = list(range(self.strategy.local_world_size()))
        print(f"local_gpus: {local_gpus}")

        while not self.stopped:
            gpus = GPUtil.getGPUs()

            for gpu in gpus: 
                if gpu.id not in local_gpus: 
                    continue
                print(gpu.id)

                empty_gpu_log = {"load": [],"memory": []}
                gpu_stats = self.monitoring_log.get(gpu.id, empty_gpu_log)

                gpu_stats["load"].append(gpu.load)
                gpu_stats["memory"].append(gpu.memoryUtil)
                self.monitoring_log[gpu.id] = gpu_stats

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        

class ProfilerTrainer(TorchTrainer):

    def train(self):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with record_function("train_func"): 
                super().train()

        # sort_string = "cuda_time_total"
        print(f"prof.key_averages():\n {prof.key_averages().table()}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--strategy", type=str, default="ddp", help="distributed strategy (default=ddp)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1.0)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()

    # Dataset creation
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
        num_workers_dataloader=1
    )
    logger = ConsoleLogger()

    trainer = TorchTrainer(
        config=training_config,
        model=model,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        checkpoint_every=args.ckpt_interval,
        logger=logger
    )
    strategy = trainer.strategy
    monitors = []

    strategy.init()
    # Launch training
    print(f"local rank: {strategy.local_rank()}, global rank: {strategy.global_rank()}")
    if strategy.local_rank() == 0: 
        print("making monitor!")
        monitor = Monitor(
                    delay=5, 
                    strategy=strategy, 
                    global_rank=strategy.global_rank()
            )
        monitors.append(monitor)
    trainer.execute(train_dataset, validation_dataset, None)
    print(f"Monitors: {monitors}")
    for monitor in monitors: 
        monitor.stop()
        print(monitor.monitoring_log)

    
    
    # print(monitor.monitoring_log)
    # global_dict = {}
    # gpu_counter = 0
    # for node_log in monitor.monitoring_log.values():  
    #     print(f"Node log:")
    #     print(node_log)
    #     for node_info in node_log.values(): 
    #         print(f"gpu_counter: {gpu_counter}")
    #         global_dict[gpu_counter] = node_info
    #         gpu_counter += 1
    # print(f"Printing global dict!")
    # print(global_dict)
    # for gpu_id, info in monitor.monitoring_log.items(): 
    #     mean_load = sum(info["load"]) / len(info["load"])
    #     print(f"Mean load for GPU {gpu_id} was {mean_load}")



if __name__ == "__main__":
    main()
