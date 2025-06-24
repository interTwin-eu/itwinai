# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import pytest

from itwinai.distributed import get_adaptive_ray_scaling_config, ray_cluster_is_running
from itwinai.torch.trainer import TorchTrainer


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
    mnist_datasets, request, strategy_fixture, named_temp_dir, mnist_net
):
    """Test TorchTrainer on MNIST with different distributed strategies."""
    training_config = dict(optimizer="sgd", loss="nllloss")
    trainer = TorchTrainer(
        model=mnist_net,
        config=training_config,
        epochs=2,
        strategy=None,
        checkpoint_every=1,
        checkpoints_location=named_temp_dir / "my_checkpoints",
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
        model=mnist_net,
        config=training_config,
        epochs=1,
        strategy=None,
        checkpoint_every=1,
        from_checkpoint=named_temp_dir / "my_checkpoints/best_model",
        checkpoints_location=named_temp_dir / "my_checkpoints_new_model",
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
def test_distributed_trainer_mnist_ray(
    mnist_datasets, strategy_name, shared_tmp_path, mnist_net
):
    """Test TorchTrainer on MNIST with different distributed strategies using Ray."""
    from ray.tune import RunConfig

    assert ray_cluster_is_running(), "Ray cluster not detected. Aborting tests"

    ray_run_config = RunConfig(storage_path=shared_tmp_path / "ray_checkpoints")

    ckpt_path = shared_tmp_path / "my_checkpoints" / strategy_name
    training_config = dict(optimizer="sgd", loss="nllloss")
    trainer = TorchTrainer(
        model=mnist_net,
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
        model=mnist_net,
        config=training_config,
        epochs=1,
        strategy=strategy_name,
        ray_scaling_config=get_adaptive_ray_scaling_config(),
        checkpoint_every=1,
        from_checkpoint=ckpt_path / "best_model",
        checkpoints_location=ckpt_path,
    )
    # Resume training
    # TODO: prevent strategy cleanup?
    trainer.execute(train_set, val_set)
