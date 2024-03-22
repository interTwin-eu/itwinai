"""
Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with an extremely simple neural network.
"""
from typing import Any
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    DDPDistributedStrategy,
    HVDDistributedStrategy,
    DSDistributedStrategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", "-s", type=str,
        choices=['ddp', 'horovod', 'deepspeed'],
        default='ddp'
    )
    parser.add_argument(
        "--shuffle_dataloader",
        action=argparse.BooleanOptionalAction
    )

    # DeepSpeed: needs to be removed
    import deepspeed
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
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


def trainer_entrypoint_fn(
        foo: Any, args: argparse.Namespace, strategy: TorchDistributedStrategy
) -> int:
    """Dummy training function. This emulates custom code developed
    by some use case.
    """
    strategy.init()
    print(f"{foo}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} "
          f"{os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")

    # Local model
    model = nn.Linear(3, 4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # Distributed model
    deepspeed_config = dict(train_batch_size=32)
    # 'config_params' key is ignored if strategy != DSDistributedStrategy
    model, optim, lr_sched = strategy.distributed(
        model, optim, lr_scheduler=None, config_params=deepspeed_config
    )

    # Data
    train_set = UniformRndDataset(x_size=3, y_size=4)
    # Distributed dataloader
    train_loader = DataLoader(
        train_set, batch_size=10, num_workers=1,
        sampler=DistributedSampler(
            train_set,
            num_replicas=strategy.dist_gwsize(),
            rank=strategy.dist_grank(),
            shuffle=args.shuffle_dataloader
        )
    )

    # Device allocated for this worker
    device = strategy.dist_device()

    for epoch in range(2):
        for (x, y) in train_loader:
            # print(f"tensor to cuda:{device}")
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            loss.backward()

            optim.step()

            if strategy.is_main_worker():
                print(f"Loss [epoch={epoch}]: {loss.item()}")
            print(f"NNLoss [epoch={epoch}]: {loss.item()}")

        # Update scheduler
        if lr_sched:
            lr_sched.step()

    print(f"<Global rank: {strategy.dist_grank()}> - TRAINING FINISHED")
    strategy.clean_up()
    return 123


if __name__ == "__main__":

    args = parse_args()

    # Instantiate Strategy
    if args.strategy == 'ddp':
        if (not torch.cuda.is_available()
                or not torch.cuda.device_count() > 1):
            raise RuntimeError('Resources unavailable')

        strategy = DDPDistributedStrategy(backend='nccl')
    elif args.strategy == 'horovod':
        strategy = HVDDistributedStrategy()
    elif args.strategy == 'deepspeed':
        strategy = DSDistributedStrategy(backend='nccl')
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented.")

    # Launch distributed training
    trainer_entrypoint_fn("foobar", args, strategy)