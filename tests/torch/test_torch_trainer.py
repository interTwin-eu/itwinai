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
from unittest.mock import MagicMock, patch

import pytest
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from itwinai.torch.trainer import TorchTrainer

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


@pytest.fixture(scope="module")
def mnist_datasets():
    """Parse MNIST datasets."""
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
    return train_set, val_set


@pytest.mark.hpc
@pytest.mark.parametrize(
    "strategy_name,strategy_fixture",
    [
        pytest.param("ddp", "ddp_strategy", marks=pytest.mark.torch_dist),
        pytest.param("deepspeed", "deepspeed_strategy", marks=pytest.mark.deepspeed_dist),
        pytest.param("horovod", "horovod_strategy", marks=pytest.mark.horovod_dist),
    ],
)
def test_distributed_trainer_mnist(mnist_datasets, request, strategy_name, strategy_fixture):
    """Test TorchTrainer on MNIST with different distributed strategies."""
    training_config = dict(optimizer="sgd", loss="nllloss")
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy=strategy_name,
        checkpoint_every=1,
    )

    strategy_instance = request.getfixturevalue(strategy_fixture)
    trainer.strategy = strategy_instance  # Patch the strategy with the fixture instance

    train_set, val_set = mnist_datasets

    # Mock strategy cleanup
    with patch.object(
        trainer.strategy, "clean_up", new=MagicMock(name="clean_up")
    ) as mock_cleanup:
        trainer.execute(train_set, val_set)

        # Check that the torch trainer is cleaning up the strategy
        mock_cleanup.assert_called_once()
