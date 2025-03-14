# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with an extremely simple neural network.
"""

import argparse
import time
from typing import Dict

import horovod.torch as hvd
import torch
from torch import nn
from torch.utils.data import Dataset

from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
    distributed_resources_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", "-s", type=str, choices=["ddp", "horovod", "deepspeed"], default="ddp"
    )
    parser.add_argument("--shuffle_dataloader", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="input batch size for training (default: 10)",
    )

    # DeepSpeed: needs to be removed
    import deepspeed

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class UniformRndDataset(Dataset):
    """Dummy torch dataset."""

    def __init__(self, x_size: int, y_size: int, len: int = 100):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.rand(self.x_size), torch.rand(self.y_size)


def training_fn(
    args: argparse.Namespace, strategy: TorchDistributedStrategy, distribute_kwargs: Dict
) -> int:
    """Dummy training function."""
    strategy.init()

    # Local model
    model = nn.Linear(3, 4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # Distributed model
    model, optim, lr_sched = strategy.distributed(
        model, optim, lr_scheduler=None, **distribute_kwargs
    )

    # Dataset
    train_set = UniformRndDataset(x_size=3, y_size=4)
    # Distributed dataloader
    train_loader = strategy.create_dataloader(
        train_set, batch_size=args.batch_size, num_workers=1, shuffle=True
    )

    # Device allocated for this worker
    device = strategy.device()

    for epoch in range(2):
        # IMPORTANT: set current epoch ID in distributed sampler
        if strategy.is_distributed:
            train_loader.sampler.set_epoch(epoch)

        for x, y in train_loader:
            # print(f"tensor to cuda:{device}")
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            loss.backward()

            optim.step()

            if strategy.is_main_worker:
                print(f"Loss [epoch={epoch}]: {loss.item()}")
            # print(f"NNLoss [epoch={epoch}]: {loss.item()}")

        # Update scheduler
        if lr_sched:
            lr_sched.step()

    time.sleep(1)
    print(f"<Global rank: {strategy.global_rank()}> - TRAINING FINISHED")
    strategy.clean_up()
    return 123


if __name__ == "__main__":
    args = parse_args()

    # Instantiate Strategy
    if not distributed_resources_available():
        print("WARNING: falling back to non-distributed strategy.")
        strategy = NonDistributedStrategy()
        distribute_kwargs = {}
    elif args.strategy == "ddp":
        strategy = TorchDDPStrategy(backend="nccl")
        distribute_kwargs = {}
    elif args.strategy == "horovod":
        strategy = HorovodStrategy()
        distribute_kwargs = dict(
            compression=hvd.Compression.none, op=hvd.Average, gradient_predivide_factor=1.0
        )
    elif args.strategy == "deepspeed":
        strategy = DeepSpeedStrategy(backend="nccl")
        distribute_kwargs = dict(
            config_params=dict(train_micro_batch_size_per_gpu=args.batch_size)
        )
    else:
        raise NotImplementedError(f"Strategy {args.strategy} is not recognized/implemented.")
    # Launch distributed training
    training_fn(args, strategy, distribute_kwargs)
