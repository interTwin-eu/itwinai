import pytest

import os
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets

from itwinai.torch.trainer import TorchTrainer
import logging

MNIST_PATH = "mnist_dataset"


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


@pytest.fixture(scope='module')
def mnist_datasets():
    """Parse MNIST datasets."""
    if not os.environ.get('MNIST_PATH'):
        logging.warning("MNIST dataset not found locally. I have to download it!")

    dataset_path = os.environ.get('MNIST_PATH', MNIST_PATH)
    train_set = datasets.MNIST(
        dataset_path,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    val_set = datasets.MNIST(
        dataset_path,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    return train_set, val_set


@pytest.mark.hpc
@pytest.mark.torch_dist
def test_distributed_trainer_ddp_mnist(mnist_datasets, ddp_strategy):
    """Test TorchTrainer on MNIST with DDP strategy."""
    training_config = dict(
        optimizer='sgd',
        loss='nllloss'
    )
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy='ddp',
        checkpoint_every=1
    )
    # Patch strategy with already initialized one to avoid re-initialization
    trainer.strategy = ddp_strategy

    train_set, val_set = mnist_datasets
    trainer.execute(train_set, val_set)


@pytest.mark.hpc
@pytest.mark.deepspeed_dist
def test_distributed_trainer_deepspeed_mnist(mnist_datasets, deepspeed_strategy):
    """Test TorchTrainer on MNIST with DeepSpeed strategy."""
    training_config = dict(
        optimizer='sgd',
        loss='nllloss'
    )
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy='deepspeed',
        checkpoint_every=1
    )
    # Patch strategy with already initialized one to avoid re-initialization
    trainer.strategy = deepspeed_strategy

    train_set, val_set = mnist_datasets
    trainer.execute(train_set, val_set)


@pytest.mark.hpc
@pytest.mark.horovod_dist
def test_distributed_trainer_horovod_mnist(mnist_datasets, horovod_strategy):
    """Test TorchTrainer on MNIST with Horovod strategy."""
    training_config = dict(
        optimizer='sgd',
        loss='nllloss'
    )
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy='horovod',
        checkpoint_every=1
    )
    # Patch strategy with already initialized one to avoid re-initialization
    trainer.strategy = horovod_strategy

    train_set, val_set = mnist_datasets
    trainer.execute(train_set, val_set)
