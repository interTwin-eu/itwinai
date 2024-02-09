"""
Show how to use DDP, Horovod and DeepSpeed strategies interchangeably.
Depending on the strategy you choose, you need to run this script with
different ad-hoc commands:

Torch DistributedDataParallel (DDP). Launch with torchrun:
>>> micromamba run -p ../../.venv-pytorch/ torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node=4 \
    ex0_multiple_torch_strategies.py -s ddp


Using a SLURM jobscript:

1. Torch DistributedDataParallel (DDP):
set STRATEGY="ddp" in ``torchrun ex0_multiple_torch_strategies.sh``
2. Horovod:
set STRATEGY="horovod" in ``torchrun ex0_multiple_torch_strategies.sh``
3. DeepSpeed:
set STRATEGY="deepspeed" in ``torchrun ex0_multiple_torch_strategies.sh``

Execute ``torchrun ex0_multiple_torch_strategies.sh`` in a slurm environment:

>>> sbatch ex0_multiple_torch_strategies.sh


"""
from typing import Any
import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from itwinai.torch.distributed import (
    TorchDistributedStrategy_old,
    DDPDistributedStrategy_old,
    HVDDistributedStrategy_old,
    DSDistributedStrategy_old
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
    return parser.parse_args()


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
        foo: Any, args: argparse.Namespace,
        strategy: TorchDistributedStrategy_old
) -> int:
    """Dummy training function. This emulates custom code developed
    by some use case.
    """
    strategy.init_backend()
    print(f"{foo}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} "
          f"{os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")

    # Local model
    model = nn.Linear(3, 4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # Distributed model
    model: nn.Module = strategy.distribute_model(model)
    optim: torch.optim.Optimizer = strategy.distribute_optimizer(optim, model)

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

    strategy.clean_up()
    return 123


if __name__ == "__main__":

    args = parse_args()

    # Instantiate Strategy
    if args.strategy == 'ddp':
        if (not torch.cuda.is_available()
                or not torch.cuda.device_count() > 1):
            raise RuntimeError('Resources unavailable')

        strategy = DDPDistributedStrategy_old(backend='nccl')
    elif args.strategy == 'horovod':
        strategy = HVDDistributedStrategy_old()
    elif args.strategy == 'deepspeed':
        strategy = DSDistributedStrategy_old(...)
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented.")

    # Launch distributed training
    trainer_entrypoint_fn("foobar", args, strategy)
