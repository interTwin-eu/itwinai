# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Elisa Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# -------------------------------------------------------------------------------------

from argparse import Namespace
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from itwinai.loggers import Logger
from itwinai.torch.loggers import ItwinaiLogger as PyTorchLightningLogger


def test_mock_experiment_log(lightning_mock_loggers):
    itwinai_logger_mock, lightning_logger = lightning_mock_loggers

    # Check that the mock is preserving default class attributes
    assert not itwinai_logger_mock.is_initialized
    assert itwinai_logger_mock.worker_rank == 0

    lightning_logger.experiment.log(
        item={"test": 10}, identifier="test_dict", kind="dict", step=1, batch_idx=1
    )
    # Check that the experiment properly initializes
    itwinai_logger_mock.create_logger_context.assert_called_once_with(rank=0)
    assert isinstance(lightning_logger.experiment, Logger)

    # Check that the log function is called properly
    itwinai_logger_mock.log.assert_called_once_with(
        item={"test": 10}, identifier="test_dict", kind="dict", step=1, batch_idx=1
    )


def test_mock_log_metrics(lightning_mock_loggers):
    """Test log_metrics method to ensure it logs metrics with the correct parameters."""
    itwinai_logger_mock, lightning_logger = lightning_mock_loggers
    metrics = {"accuracy": 0.95, "loss": 0.1}
    step = 1
    lightning_logger.log_metrics(metrics=metrics, step=step)
    # Check that the log function is called correctly
    for key, value in metrics.items():
        itwinai_logger_mock.log.assert_any_call(
            item=value, identifier=key, kind="metric", step=step
        )


def test_moc_log_hyperparams(lightning_mock_loggers):
    """Test log_hyperparams method to ensure it logs hyperparameters correctly."""
    itwinai_logger_mock, lightning_logger = lightning_mock_loggers

    dict_params = {"learning_rate": 0.001, "batch_size": 32}
    lightning_logger.log_hyperparams(params=dict_params)
    itwinai_logger_mock.save_hyperparameters.assert_called_once_with(dict_params)

    itwinai_logger_mock.reset_mock()

    namespace_params = Namespace(**dict_params)
    lightning_logger.log_hyperparams(params=namespace_params)
    itwinai_logger_mock.save_hyperparameters.assert_called_once_with(dict_params)


def test_mock_after_save_checkpoint(lightning_mock_loggers):
    """Test after_save_checkpoint method in PyTorchLightningLogger."""
    _, lightning_logger = lightning_mock_loggers

    checkpoint_callback = MagicMock(spec=ModelCheckpoint)
    checkpoint_callback.save_top_k = -1
    checkpoint_callback.best_model_path = Path("/path/to/best_model.ckpt")
    assert lightning_logger._checkpoint_callback is None

    # Scenario 1:
    lightning_logger._log_model = "all"
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback is None

    # Scenario 2: Should checkpoint
    lightning_logger._log_model = True
    checkpoint_callback.save_top_k = -1
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback is None

    # Scenario 3:
    lightning_logger._log_model = False
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback is None

    # Scenario 4:
    lightning_logger._log_model = True
    checkpoint_callback.save_top_k = 5
    lightning_logger.after_save_checkpoint(checkpoint_callback)
    assert lightning_logger._checkpoint_callback == checkpoint_callback


@pytest.mark.parametrize(
    "itwinai_logger",
    [
        "console_logger",
        "mlflow_logger",
        "wandb_logger",
        "tensorboard_logger_torch",
        "prov4ml_logger",
        "loggers_collection",
    ],
)
def test_experiment_and_finalize(itwinai_logger, request):
    itwinai_logger_instance = request.getfixturevalue(itwinai_logger)
    lightning_logger = PyTorchLightningLogger(itwinai_logger=itwinai_logger_instance)

    def create_logger_side_effect(*args, **kwargs):
        # Mimic the real method behavior of doing nothing if already initialized
        if not itwinai_logger_instance.is_initialized:
            itwinai_logger_instance.is_initialized = True
        return

    with patch.object(
        itwinai_logger_instance, "create_logger_context", side_effect=create_logger_side_effect
    ) as mock_create_context:
        assert not lightning_logger.itwinai_logger.is_initialized
        assert lightning_logger.name is None
        assert lightning_logger.version is None

        # Trigger the calls to `create_logger_context`
        experiment = lightning_logger.experiment
        mock_create_context.assert_called_once_with(rank=0)
        assert isinstance(experiment, Logger)


@pytest.mark.parametrize(
    "itwinai_logger",
    [
        "console_logger",
        "mlflow_logger",
        "wandb_logger",
        "tensorboard_logger_torch",
        "prov4ml_logger",
        "loggers_collection",
    ],
)
def test_log_metrics_and_hyperparams(itwinai_logger, request):
    itwinai_logger_instance = request.getfixturevalue(itwinai_logger)
    lightning_logger = PyTorchLightningLogger(itwinai_logger=itwinai_logger_instance)

    with (
        patch.object(itwinai_logger_instance, "create_logger_context"),
        patch.object(itwinai_logger_instance, "log"),
        patch.object(itwinai_logger_instance, "save_hyperparameters"),
    ):
        lightning_logger.log_metrics({"loss": 0.2, "accuracy": 0.99})
        expected_calls = [
            call(item=0.2, identifier="loss", kind="metric", step=None),
            call(item=0.99, identifier="accuracy", kind="metric", step=None),
        ]
        itwinai_logger_instance.log.assert_has_calls(expected_calls)

        dict_params = {"learning_rate": 0.001, "batch_size": 32}
        lightning_logger.log_hyperparams(params=dict_params)
        itwinai_logger_instance.save_hyperparameters.assert_called_once_with(dict_params)


@pytest.mark.parametrize(
    "itwinai_logger",
    [
        "console_logger",
        "mlflow_logger",
        "wandb_logger",
        "tensorboard_logger_torch",
        "prov4ml_logger",
        "loggers_collection",
    ],
)
def test_save_after_checkpoint(itwinai_logger, request):
    itwinai_logger_instance = request.getfixturevalue(itwinai_logger)
    lightning_logger = PyTorchLightningLogger(
        itwinai_logger=itwinai_logger_instance, log_model="all"
    )
    checkpoint_callback = MagicMock(spec=ModelCheckpoint)
    checkpoint_callback.best_model_path = Path("/path/to/checkpoint_1.ckpt")

    with (
        patch.object(itwinai_logger_instance, "create_logger_context"),
        patch.object(itwinai_logger_instance, "log"),
        patch(
            "itwinai.torch.loggers._scan_checkpoints",
            return_value=[
                (1, "path/to/checkpoint_1.ckpt", 100, "_"),
                (2, "path/to/checkpoint_2.ckpt", 101, "_"),
            ],
        ),
    ):
        lightning_logger.after_save_checkpoint(checkpoint_callback)
        assert lightning_logger._checkpoint_callback is None

        expected_calls = [
            call(
                item="path/to/checkpoint_1.ckpt",
                identifier="checkpoint_1",
                kind="artifact",
            ),
            call(item=ANY, identifier="checkpoint_1", kind="artifact"),
            call(
                item="path/to/checkpoint_2.ckpt",
                identifier="checkpoint_2",
                kind="artifact",
            ),
            call(item=ANY, identifier="checkpoint_2", kind="artifact"),
        ]
        itwinai_logger_instance.log.assert_has_calls(expected_calls)
