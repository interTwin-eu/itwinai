import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from strategy import Strategy, DDPStrategy
from launcher import DummyTorchElasticLauncher
from cluster import (
    LocalEnvironment, SLURMEnvironment,
    TorchElasticEnvironment
)


class UniformRndDataset(Dataset):
    def __init__(self, x_size: int, y_size: int, len: int = 100):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.rand(self.x_size), torch.rand(self.y_size)


def trainer_entrypoint_fn(a, strategy: Strategy):
    """Dummy training function."""
    strategy.setup()
    print(f"{a}: {os.environ.get('RANK')} {os.environ.get('LOCAL_RANK')} "
          f"{os.environ.get('MASTER_ADDR')} {os.environ.get('MASTER_PORT')}")

    # Local model
    model = nn.Linear(3, 4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # Distributed model
    model: nn.Module = strategy.distribute_model(model)
    optim: torch.optim.Optimizer = strategy.distribute_optimizer(optim)

    # Data
    train_set = UniformRndDataset(x_size=3, y_size=4)
    train_loader = DataLoader(train_set, batch_size=10, num_workers=1)
    # Distributed dataloader
    train_loader: DataLoader = strategy.distribute_dataloader(train_loader)

    for epoch in range(2):
        for (x, y) in train_loader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)

            optim.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()

            if strategy.is_main_worker():
                print(f"Loss [epoch={epoch}]: {loss.item()}")

    strategy.teardown()
    return 123


STRATEGY = 'ddp'

RUN_ID = "my_run_id"
MIN_NODES = 1
MAX_NODES = 1
NPROC_PRE_NODE = 4
MAX_RESTARTS = 2

if __name__ == "__main__":
    # STRATEGY BUILDER

    # Instantiate ClusterEnv
    if SLURMEnvironment.detect():
        cluster = SLURMEnvironment()
    elif TorchElasticEnvironment.detect():
        cluster = TorchElasticEnvironment()
    elif LocalEnvironment.detect():
        cluster = LocalEnvironment()
    else:
        raise NotImplementedError("Unrecognized cluster env")

    print(cluster)

    # Instantiate Launcher
    launcher = DummyTorchElasticLauncher(
        cluster=cluster,
        n_workers_per_node=NPROC_PRE_NODE,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES
    )

    # Instantiate Strategy
    if (STRATEGY == 'ddp'
        and torch.cuda.is_available()
            and torch.cuda.device_count() > 1):
        strategy = DDPStrategy(cluster=cluster, backend='nccl')
    else:
        raise NotImplementedError

    # CLIENT CODE
    # Launch training from launcher
    launcher.run(func=trainer_entrypoint_fn, args=("foobar", strategy))
