# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset

from itwinai.torch.trainer import TorchTrainer


# Sample training configuration mock
def get_mock_config():
    return {
        "optim_lr": 0.001,
        "optimizer": "adam",
        "batch_size": 5,
        "loss": "mse",
        "lr_scheduler": "linear",
    }


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


def to_cpu_recursive(state_dict):
    from collections import OrderedDict

    """Recursively moves all tensors in a state_dict to CPU while preserving OrderedDict
    structure.
    """
    if isinstance(state_dict, OrderedDict):
        return OrderedDict((k, to_cpu_recursive(v)) for k, v in state_dict.items())
    elif isinstance(state_dict, dict):  # Handle nested dicts
        return {k: to_cpu_recursive(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, torch.Tensor):  # Move tensors to CPU
        return state_dict.to("cpu")
    return state_dict  # Return unchanged for non-tensor values


def equal_models(model1: torch.nn.Module, model2: torch.nn.Module) -> bool:
    # Compare the state dictionaries
    state_dict1 = to_cpu_recursive(model1.state_dict())
    state_dict2 = to_cpu_recursive(model2.state_dict())

    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Mismatch in parameter: {key}")
            return False

    return True


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


@pytest.mark.parametrize(
    "strategy_fixture",
    [
        pytest.param(None),
        pytest.param("ddp_strategy", marks=[pytest.mark.hpc, pytest.mark.torch_dist]),
        pytest.param(
            "deepspeed_strategy", marks=[pytest.mark.hpc, pytest.mark.deepspeed_dist]
        ),
        pytest.param("horovod_strategy", marks=[pytest.mark.hpc, pytest.mark.horovod_dist]),
    ],
)
def test_checkpoint_loading(strategy_fixture: str | None, named_temp_dir: Path, request):
    """Test loading of checkpoints that were generated by some distributed strategy.
    This is ignoring Ray strategies.
    """
    model = torch.nn.Linear(10, 1)
    untrained_model = copy.deepcopy(model)
    ckpt_path = named_temp_dir / "checkpoint"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"Failed to create checkpoints path at {ckpt_path.resolve()}"

    train_dataset = DummyDataset()
    validation_dataset = DummyDataset()

    # Create cehckpoint throigh the TorchTrainer execution
    trainer = TorchTrainer(
        config=get_mock_config(),
        epochs=2,
        model=model,
        strategy=None,
        checkpoint_every=1,
        checkpoints_location=ckpt_path,
        random_seed=42,
    )

    if strategy_fixture:
        # Patch trainer's strategy with strategy fixture
        trainer.strategy = request.getfixturevalue(strategy_fixture)

    # Mock some methods to skip training
    best_val_loss = 0.5
    trainer.validation_epoch = MagicMock()
    # Set list of return values for successive calls -- simulate reduction in val loss
    trainer.validation_epoch.side_effect = [torch.tensor(0.9), torch.tensor(best_val_loss)]

    # Run training: generate checkpoints
    # Mock strategy cleanup -- IMPORTANT, otherwise the trainer will mess up with the strategy
    # fixture
    with patch.object(
        trainer.strategy, "clean_up", new=MagicMock(name="clean_up")
    ) as mock_cleanup:
        trainer.execute(train_dataset=train_dataset, validation_dataset=validation_dataset)

        # Check that the torch trainer is cleaning up the strategy
        mock_cleanup.assert_called_once()

    # Check that the model was actually trained
    assert not equal_models(trainer.model, untrained_model)

    best_ckpt_path = ckpt_path / "best_model"
    assert best_ckpt_path.exists(), (
        f"Could not find best checkpoint path at {best_ckpt_path.resolve()}. "
        f"Checkpoints location contains: {list(ckpt_path.glob('**/**'))}"
    )

    # Create TorchTrainer instance with checkpoint
    model = torch.nn.Linear(10, 1)
    target_ckpt = best_ckpt_path
    trainer = TorchTrainer(
        config=get_mock_config(),
        epochs=2,
        model=model,
        strategy=None,
        from_checkpoint=target_ckpt,
    )

    trainer.create_dataloaders = MagicMock()
    trainer.train = MagicMock()
    # Run training: force checkpoint to be loaded
    # Mock strategy cleanup -- IMPORTANT, otherwise the trainer will mess up with the strategy
    # fixture
    with patch.object(
        trainer.strategy, "clean_up", new=MagicMock(name="clean_up")
    ) as mock_cleanup:
        trainer.execute(train_dataset=MagicMock(), validation_dataset=MagicMock())

        # Check that the torch trainer is cleaning up the strategy
        mock_cleanup.assert_called_once()

    # Manually simulate the loading of checkpoint
    state = torch.load(target_ckpt / "state.pt", weights_only=True)
    model_state_dict = torch.load(target_ckpt / "model.pt", weights_only=True)
    optimizer_state_dict = state["optimizer_state_dict"]
    lr_scheduler_state_dict = state["lr_scheduler_state_dict"]
    torch_rng_state = state["torch_rng_state"]
    random_seed = state["random_seed"]
    epoch = state["epoch"]
    best_validation_loss = None
    if state["best_validation_metric"]:
        best_validation_loss = state["best_validation_metric"]
    loaded_model = torch.nn.Linear(10, 1)
    loaded_model.load_state_dict(model_state_dict, strict=False)
    loaded_optim = Adam(loaded_model.parameters())
    loaded_lr_scheduler = LinearLR(loaded_optim)
    # Remember to load optim state dict after the scheduler has already been instantiated
    loaded_optim.load_state_dict(optimizer_state_dict)
    loaded_lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    # Best validation loss
    assert best_validation_loss == best_val_loss
    assert trainer.best_validation_metric == best_val_loss, (
        "Best validation loss was not correctly restored"
    )

    # Validate that state is properly loaded -- restart from the following epoch
    assert trainer.current_epoch == epoch + 1, "Epoch should be restored from checkpoint"

    # Validate model parameters
    assert equal_models(trainer.model, loaded_model), "Model checkpoint not loaded correctly"

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


if __name__ == "__main__":
    import torch
    from torch.nn import Linear

    model1 = Linear(in_features=10, out_features=1, bias=True)
    model2 = Linear(in_features=10, out_features=1, bias=True)
    model2.load_state_dict(model1.state_dict())
    model1 = model1.to("cuda")
    print(equal_models(model1, model2))
