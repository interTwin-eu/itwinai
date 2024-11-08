from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from itwinai.loggers import Logger
from itwinai.torch.loggers import PyTorchLightningLogger


@pytest.fixture
def lightning_loggers():
    """Setup a generic PyTorchLightningLogger with mock loggers for testing."""
    itwinai_logger_mock = MagicMock()
    lightning_logger = PyTorchLightningLogger(itwinai_logger=itwinai_logger_mock)

    return itwinai_logger_mock, lightning_logger


def test_lightning_experiment_property(lightning_loggers):
    """Test that the experiment property initializes
    the logger context and returns the logger."""
    itwinai_logger_mock, lightning_logger = lightning_loggers
    experiment = lightning_logger.experiment
    assert isinstance(experiment, Logger)
    itwinai_logger_mock.create_logger_context.assert_called_once_with(rank=0)


def test_lightning_log_metrics(lightning_loggers):
    """Test log_metrics method to ensure it logs metrics with the correct parameters."""
    itwinai_logger_mock, lightning_logger = lightning_loggers
    metrics = {"accuracy": 0.95, "loss": 0.1}
    step = 1
    lightning_logger.log_metrics(metrics=metrics, step=step)
    for key, value in metrics.items():
        itwinai_logger_mock.log.assert_any_call(
            item=value, identifier=key, kind='metric', step=step)


def test_lightning_log_hyperparams(lightning_loggers):
    """Test log_hyperparams method to ensure it logs hyperparameters correctly."""
    itwinai_logger_mock, lightning_logger = lightning_loggers
    dict_params = {"learning_rate": 0.001, "batch_size": 32}
    namespace_params = Namespace(**dict_params)

    lightning_logger.log_hyperparams(params=dict_params)
    itwinai_logger_mock.save_hyperparameters.assert_called_once_with(dict_params)

    itwinai_logger_mock.reset_mock()
    lightning_logger.log_hyperparams(params=namespace_params)
    itwinai_logger_mock.save_hyperparameters.assert_called_once_with(dict_params)


def test_lightning_after_save_checkpoint_1(lightning_loggers):
    """Test after_save_checkpoint method in PyTorchLightningLogger."""
    itwinai_logger_mock, lightning_logger = lightning_loggers

    checkpoint_callback = MagicMock(spec=ModelCheckpoint)
    checkpoint_callback.save_top_k = -1
    checkpoint_callback.best_model_path = "/path/to/best_model.ckpt"
    assert lightning_logger._checkpoint_callback is None

    # Scenario 1:
    lightning_logger._log_model = "all"
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback == None

    # Scenario 2: Should checkpoint
    lightning_logger._log_model = True
    checkpoint_callback.save_top_k = -1
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback == None

    # Scenario 3:
    lightning_logger._log_model = False
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback is None

    # Scenario 4:
    lightning_logger._log_model = True
    checkpoint_callback.save_top_k = 5
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback == checkpoint_callback


def test_lightning_experiment_log(lightning_loggers):
    itwinai_logger_mock, lightning_logger = lightning_loggers

    lightning_logger.experiment.log(
        item={"test": 10},
        identifier="test_dict",
        kind="dict",
        step=1,
        batch_idx=1
    )
    itwinai_logger_mock.log.assert_called_once_with(
        item={"test": 10},
        identifier="test_dict",
        kind="dict",
        step=1,
        batch_idx=1
    )
