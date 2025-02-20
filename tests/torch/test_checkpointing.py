from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset

from itwinai.torch.trainer import TorchTrainer


# Sample training configuration mock
def get_mock_config():
    return {"optim_lr": 0.001, "optimizer": "adam", "loss": "mse", "lr_scheduler": "linear"}


# Distributed strategies to test
STRATEGIES = ["ddp", "deepspeed", "horovod"]
STRATEGIES = ["ddp"]


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx].type(torch.float), self.targets[idx].type(torch.float)


def assert_equal_optimizer_states(state1, state2):
    """Compares two optimizer state dictionaries."""

    """Compares two optimizer state dictionaries with detailed assertions."""
    assert state1.keys() == state2.keys(), "Optimizer state keys mismatch"

    # Compare param_groups excluding 'params' because indexes might differ
    for key in state1["param_groups"][0]:
        if key != "params":
            assert state1["param_groups"][0][key] == state2["param_groups"][0][key], (
                f"Mismatch in param_groups key: {key}"
            )

    # Ensure both have the same parameter keys
    assert state1["state"].keys() == state2["state"].keys(), (
        "Parameter keys in optimizer state do not match"
    )

    # Compare each parameter state
    for param_id in state1["state"]:
        assert param_id in state2["state"], (
            f"Parameter {param_id} missing in second optimizer state"
        )

        for subkey in state1["state"][param_id]:
            tensor1 = state1["state"][param_id][subkey]
            tensor2 = state2["state"][param_id].get(subkey)

            assert tensor2 is not None, (
                f"Missing key '{subkey}' for param {param_id} in second state"
            )

            assert torch.allclose(tensor1, tensor2, atol=1e-6), (
                f"Mismatch in {subkey} for param {param_id}"
            )


def test_inexistent_checkpoint():
    # Test unexisting checkpoint directory
    with pytest.raises(RuntimeError) as err:
        TorchTrainer(
            config=get_mock_config(),
            epochs=2,
            model=None,
            strategy="ddp",
            from_checkpoint="/my/checkpoint/dir",
            checkpoint_every=1,
        )
        assert "checkpoint is not found" in err


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_checkpoint_loading(strategy, tmp_path):
    model = torch.nn.Linear(10, 1)
    ckpt_path = tmp_path / "checkpoint"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_dataset = DummyDataset()
    validation_dataset = DummyDataset()

    # Create cehckpoint throigh the TorchTrainer execution
    trainer = TorchTrainer(
        config=get_mock_config(),
        epochs=2,
        model=model,
        strategy=strategy,
        checkpoint_every=1,
        checkpoints_location=ckpt_path,
        random_seed=42,
    )

    # Mock some methods to skip training
    best_val_loss = 0.5
    trainer.validation_epoch = MagicMock()
    # Set list of return values for successive calls -- simulate reduction in val loss
    trainer.validation_epoch.side_effect = [torch.tensor(0.9), torch.tensor(best_val_loss)]

    # Run training -- generate checkpoints
    trainer.execute(train_dataset=train_dataset, validation_dataset=validation_dataset)

    # Create TorchTrainer instance with checkpoint
    target_ckpt = ckpt_path / "best_model"
    trainer = TorchTrainer(
        config=get_mock_config(),
        epochs=2,
        model=model,
        strategy=strategy,
        from_checkpoint=target_ckpt,
    )

    trainer.create_dataloaders = MagicMock()
    trainer.train = MagicMock()
    # Run training -- force checkpoint to be loaded
    trainer.execute(train_dataset=MagicMock(), validation_dataset=MagicMock())

    # Manually simulate the loading of checkpoint
    state = torch.load(target_ckpt / "state.pt")
    model_state_dict = torch.load(target_ckpt / "model.pt")
    optimizer_state_dict = state["optimizer_state_dict"]
    lr_scheduler_state_dict = state["lr_scheduler_state_dict"]
    torch_rng_state = state["torch_rng_state"]
    random_seed = state["random_seed"]
    epoch = state["epoch"]
    best_validation_loss = None
    if state["best_validation_loss"]:
        best_validation_loss = state["best_validation_loss"]
    loaded_model = torch.nn.Linear(10, 1)
    loaded_model.load_state_dict(model_state_dict)
    loaded_optim = Adam(loaded_model.parameters())
    loaded_lr_scheduler = LinearLR(loaded_optim)
    # Remember to load optim state dict after the scheduler has already been instantiated
    loaded_optim.load_state_dict(optimizer_state_dict)
    loaded_lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    raise ValueError(loaded_lr_scheduler.state_dict())

    # Best validation loss
    assert best_validation_loss == best_val_loss
    assert trainer.best_validation_loss == best_val_loss, (
        "Best validation loss was not correctly restored"
    )

    # Validate that state is properly loaded
    assert trainer.epoch == epoch, "Epoch should be restored from checkpoint"

    # Validate model parameters
    for p1, p2 in zip(trainer.model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters do not match after loading checkpoint"

    # Validate optimizer state
    assert trainer.optimizer is not None, "Optimizer should be initialized"
    assert_equal_optimizer_states(loaded_optim.state_dict(), trainer.optimizer.state_dict())

    # Validate LR scheduler state
    assert trainer.lr_scheduler.state_dict() == loaded_lr_scheduler.state_dict(), (
        "LR scheduler state mismatch"
    )

    # Validate RNG state
    assert trainer.random_seed == random_seed, "Random seed was not restored"
    assert torch.equal(torch_rng_state, trainer.torch_rng_state), "Torch RNG state mismatch"
