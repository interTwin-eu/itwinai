# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset

from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


MNIST_PATH = "mnist_dataset"


@pytest.fixture(scope="package")
def ddp_strategy() -> Generator[TorchDistributedStrategy, None, None]:
    """Instantiate Torch's DistributedDataParallel strategy."""
    strategy = TorchDDPStrategy(backend="nccl" if torch.cuda.is_available() else "gloo")
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope="package")
def deepspeed_strategy() -> Generator[DeepSpeedStrategy, None, None]:
    """Instantiate DeepSpeed strategy."""
    strategy = DeepSpeedStrategy(backend="nccl" if torch.cuda.is_available() else "gloo")
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope="package")
def horovod_strategy() -> Generator[HorovodStrategy, None, None]:
    """Instantiate Horovod strategy."""
    strategy = HorovodStrategy()
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope="function")
def shared_tmp_path():
    """Return the Path to a shared filesystem that all nodes can access.
    This is good for Ray-based Trainer because it gets broadcasted to all workers by Ray.
    /tmp location is usually a local filesystem. Uses as a prefix SHARED_FS_PATH
    env variable, but if that's not set it falls back to /tmp.
    """
    if not os.environ.get("SHARED_FS_PATH"):
        logging.warning(
            "SHARED_FS_PATH env var not set! Falling back to /tmp, but this could cause "
            "problems as this fixture should return the path to a location reachable by all "
            "nodes (/tmp usually isn't)."
        )
    else:
        Path(os.environ.get("SHARED_FS_PATH")).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SHARED_FS_PATH", "/tmp")) as tmp_path:
        yield Path(tmp_path)


@pytest.fixture(scope="function")
def named_temp_dir(request):
    """Create a temporay directory for a test case with the name of the test.
    This is useful when you want a tempdir which has a stable path across multiple workers, but
    you are not relying on Ray Trainer to broadcast a single path to all workers.
    """
    user_id = os.getuid()  # Get Linux user ID to avoid collisions among users
    test_name = request.node.name  # Get the name of the calling test

    base_path = Path(os.environ.get("SHARED_FS_PATH", "/tmp"))

    temp_dir = base_path / str(user_id) / test_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir

    # Cleanup after the test, but wait a bit for all the workers to have accessed the files
    # before deleting them
    time.sleep(5)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def mnist_datasets():
    """Parse MNIST datasets."""

    import logging
    import os

    from torchvision import datasets, transforms

    if not os.environ.get("MNIST_PATH"):
        logging.warning("MNIST dataset not found locally. I have to download it!")

    dataset_path = os.environ.get("MNIST_PATH", MNIST_PATH)
    train_set = datasets.MNIST(
        dataset_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    val_set = datasets.MNIST(
        dataset_path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Downsample datasets
    train_set = Subset(train_set, range(500))
    val_set = Subset(val_set, range(500))
    return train_set, val_set


@pytest.fixture(scope="function")
def mnist_net():
    return Net()
