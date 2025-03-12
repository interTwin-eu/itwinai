"""Test Ray Tune integration in the itwinai TorchTrainer."""

import time
from types import MethodType

import numpy as np
import pytest
from ray import tune
from ray.train import RunConfig, ScalingConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import AsyncHyperBandScheduler

from itwinai.torch.distributed import NonDistributedStrategy
from itwinai.torch.trainer import TorchTrainer

# from itwinai.distributed import get_adaptive_ray_scaling_config


@pytest.mark.hpc
@pytest.mark.ray_dist
def test_tuning_mnist_ray(mnist_datasets, shared_tmp_path, mnist_net):
    """Test TorchTrainer on MNIST with different distributed strategies using Ray."""

    def dummy_train(self: TorchTrainer):
        """Dummy training function"""

        # Check that the strategy is not running distributed ML
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

    # # Restore training from checkpoint
    # trainer = TorchTrainer(
    #     model=mnist_net,
    #     config=training_config,
    #     epochs=2,
    #     # The strategy should fallback to non-distributed strategy if
    #     # not enough resources are granted to each trial
    #     strategy="ddp",
    #     ray_scaling_config=scaling_config,
    #     checkpoint_every=1,
    #     from_checkpoint=ckpt_path / "best_model",
    #     checkpoints_location=ckpt_path,
    # )
    # # Resume training
    # # TODO: prevent strategy cleanup?
    # trainer.execute(train_set, val_set)
