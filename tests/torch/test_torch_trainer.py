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

from itwinai.distributed import get_adaptive_ray_scaling_config
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


@pytest.mark.parametrize(
    "strategy_fixture",
    [
        pytest.param(None),  # NonDistributedStrategy
        pytest.param(
            "ddp_strategy",
            marks=[pytest.mark.torch_dist, pytest.mark.hpc],
        ),
        pytest.param(
            "deepspeed_strategy",
            marks=[pytest.mark.deepspeed_dist, pytest.mark.hpc],
        ),
        pytest.param(
            "horovod_strategy",
            marks=[pytest.mark.horovod_dist, pytest.mark.hpc],
        ),
    ],
)
def test_distributed_trainer_mnist(
    mnist_datasets,
    request,
    strategy_fixture,
    tmp_path,
):
    """Test TorchTrainer on MNIST with different distributed strategies."""
    training_config = dict(optimizer="sgd", loss="nllloss")
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy=None,
        checkpoint_every=1,
        checkpoints_location=tmp_path / "my_checkpoints",
    )
    if strategy_fixture:
        # Override when the strategy is supposed to be distributed
        trainer.strategy = request.getfixturevalue(strategy_fixture)

    train_set, val_set = mnist_datasets

    # Mock strategy cleanup -- IMPORTANT, otherwise the trainer will mess up with the strategy
    # fixture
    with patch.object(
        trainer.strategy, "clean_up", new=MagicMock(name="clean_up")
    ) as mock_cleanup:
        trainer.execute(train_set, val_set)

        # Check that the torch trainer is cleaning up the strategy
        mock_cleanup.assert_called_once()

    # Restore training from checkpoint
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy=None,
        checkpoint_every=1,
        from_checkpoint=tmp_path / "my_checkpoints/best_model",
        checkpoints_location=tmp_path / "my_checkpoints",
    )
    # Mock strategy cleanup -- IMPORTANT, otherwise the trainer will mess up with the strategy
    # fixture
    with patch.object(
        trainer.strategy, "clean_up", new=MagicMock(name="clean_up")
    ) as mock_cleanup:
        trainer.execute(train_set, val_set)

        # Check that the torch trainer is cleaning up the strategy
        mock_cleanup.assert_called_once()


@pytest.mark.hpc
@pytest.mark.ray_dist
@pytest.mark.parametrize(
    "strategy_name",
    [
        pytest.param("ddp"),
        pytest.param("deepspeed"),
        pytest.param("horovod"),
    ],
)
def test_distributed_trainer_mnist_ray(mnist_datasets, strategy_name, shared_tmp_path):
    """Test TorchTrainer on MNIST with different distributed strategies using Ray."""
    from ray.train import RunConfig

    ray_run_config = RunConfig(storage_path=shared_tmp_path / "ray_checkpoints")

    # from pathlib import Path

    # ckpt_path = (
    #     Path("/p/project1/intertwin/bunino1/itwinai/tests/torch/checkpoints") / strategy_name
    # )

    ckpt_path = shared_tmp_path / "my_checkpoints" / strategy_name
    training_config = dict(optimizer="sgd", loss="nllloss")
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy=strategy_name,
        ray_scaling_config=get_adaptive_ray_scaling_config(),
        ray_run_config=ray_run_config,
        checkpoint_every=1,
        checkpoints_location=ckpt_path,
    )

    train_set, val_set = mnist_datasets

    # Train
    # TODO: prevent strategy cleanup?
    trainer.execute(train_set, val_set)

    # Restore training from checkpoint
    trainer = TorchTrainer(
        model=Net(),
        config=training_config,
        epochs=2,
        strategy=strategy_name,
        ray_scaling_config=get_adaptive_ray_scaling_config(),
        checkpoint_every=1,
        from_checkpoint=ckpt_path / "best_model",
        checkpoints_location=ckpt_path,
    )
    # Resume training
    # TODO: prevent strategy cleanup?
    trainer.execute(train_set, val_set)
