"""Test Ray Tune integration in the itwinai TorchTrainer."""

import time
from types import MethodType

import numpy as np
import pytest
import ray
from ray import tune
from ray.train import ScalingConfig
from ray.tune import RunConfig, TuneConfig
from ray.tune.schedulers import AsyncHyperBandScheduler

from itwinai.distributed import ray_cluster_is_running
from itwinai.torch.trainer import TorchTrainer


@pytest.mark.hpc
@pytest.mark.ray_dist
def test_tuning_mnist_ray(mnist_datasets, shared_tmp_path, mnist_net):
    """Test HPO for TorchTrainer on MNIST."""

    assert ray_cluster_is_running(), "Ray cluster not detected. Aborting tests"

    def dummy_train(self: TorchTrainer):
        """Dummy training function"""

        # Check that the strategy is NOT running distributed ML
        assert self.strategy.global_rank() == 0, (
            "Global rank is not 0. Does the strategy think it's distributed?"
        )
        assert self.strategy.global_world_size() == 1, (
            "Global world size is not 1. Does the strategy think it's distributed?"
        )
        # assert isinstance(self.strategy, NonDistributedStrategy), (
        #     "training should not be distributed"
        # )

        # Check that the hyperparameters are used
        assert self.config.batch_size != INITIAL_BATCH_SIZE, "Batch size was not overridden"

        for i in range(1, self.epochs + 1):
            time.sleep(0.1)
            # Simulate the reporting of validation loss at the end of an epoch
            self.ray_report(
                metrics={"loss": 2 / i + np.random.randn()},
            )

    INITIAL_BATCH_SIZE = 17
    search_space = {"batch_size": tune.qrandint(128, 1024, 128)}
    tune_config = TuneConfig(
        metric="loss",
        mode="min",
        search_alg=None,  # Random search
        scheduler=AsyncHyperBandScheduler(max_t=10, grace_period=3, reduction_factor=4),
        num_samples=2,
    )
    ray_run_config = RunConfig(storage_path=shared_tmp_path / "ray_checkpoints")
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8},
    )

    ckpt_path = shared_tmp_path / "my_checkpoints"
    training_config = dict(optimizer="sgd", loss="nllloss", batch_size=INITIAL_BATCH_SIZE)
    trainer = TorchTrainer(
        model=mnist_net,
        config=training_config,
        epochs=50,
        # The strategy should fallback to non-distributed strategy if
        # not enough resources are granted to each trial
        strategy="ddp",
        ray_scaling_config=scaling_config,
        ray_run_config=ray_run_config,
        ray_search_space=search_space,
        ray_tune_config=tune_config,
        checkpoint_every=1,
        checkpoints_location=ckpt_path,
    )

    train_set, val_set = mnist_datasets

    # Monkey patch train function
    trainer.train = MethodType(dummy_train, trainer)

    # Train
    # TODO: prevent strategy cleanup?
    trainer.execute(train_set, val_set)

    # Check if at least one trial failed
    for result in trainer.tune_result_grid:
        assert not result.error, str(result.error)


@pytest.mark.hpc
@pytest.mark.ray_dist
def test_tuning_dist_ml_mnist_ray(mnist_datasets, shared_tmp_path, mnist_net):
    """Test HPO + distributed ML for each trial, for TorchTrainer on MNIST.
    Note that this test requires at least 4 GPUs to be executed.
    """

    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    num_trials = 2
    gpus_per_trial = num_gpus // num_trials

    # TODO: remove this condition in future PRs!
    if gpus_per_trial <= 1:
        return

    assert gpus_per_trial > 1, "Not enough resources to run distributed ML under HPO."

    def dummy_train(self: TorchTrainer):
        """Dummy training function"""

        # Check that the strategy is running distributed ML
        assert self.strategy.global_world_size() == gpus_per_trial, (
            "Global world size is not 1. Does the strategy think it's distributed?"
        )

        # Check that the hyperparameters are used
        assert self.config.batch_size != INITIAL_BATCH_SIZE, "Batch size was not overridden"

        for i in range(1, self.epochs + 1):
            time.sleep(0.1)
            # Simulate the reporting of validation loss at the end of an epoch
            self.ray_report(
                metrics={"loss": 2 / i + np.random.randn()},
            )

    INITIAL_BATCH_SIZE = 17
    search_space = {"batch_size": tune.choice([32, 50, 64])}
    tune_config = TuneConfig(
        metric="loss",
        mode="min",
        search_alg=None,  # Random search
        scheduler=AsyncHyperBandScheduler(max_t=10, grace_period=3, reduction_factor=4),
        num_samples=num_trials,
    )
    ray_run_config = RunConfig(storage_path=shared_tmp_path / "ray_checkpoints")
    scaling_config = ScalingConfig(
        num_workers=gpus_per_trial,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_trial, "CPU": 8},
    )

    ckpt_path = shared_tmp_path / "my_checkpoints"
    training_config = dict(optimizer="sgd", loss="nllloss", batch_size=INITIAL_BATCH_SIZE)
    trainer = TorchTrainer(
        model=mnist_net,
        config=training_config,
        epochs=50,
        # The strategy should fallback to non-distributed strategy if
        # not enough resources are granted to each trial
        strategy="ddp",
        ray_scaling_config=scaling_config,
        ray_run_config=ray_run_config,
        ray_search_space=search_space,
        ray_tune_config=tune_config,
        checkpoint_every=1,
        checkpoints_location=ckpt_path,
    )

    train_set, val_set = mnist_datasets

    # Monkey patch train function
    trainer.train = MethodType(dummy_train, trainer)

    # Train
    # TODO: prevent strategy cleanup?
    trainer.execute(train_set, val_set)

    # Check if at least one trial failed
    for result in trainer.tune_result_grid:
        assert not result.error, str(result.error)
