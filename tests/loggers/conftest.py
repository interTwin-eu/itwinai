# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# -------------------------------------------------------------------------------------

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from itwinai.loggers import (
    ConsoleLogger,
    Logger,
    LoggersCollection,
    MLFlowLogger,
    Prov4MLLogger,
    TensorBoardLogger,
    WandBLogger,
)
from itwinai.torch.loggers import ItwinaiLogger as PyTorchLightningLogger


@pytest.fixture
def console_logger():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "console/test_mllogs"
        yield ConsoleLogger(savedir=save_dir, log_freq=1)


@pytest.fixture(scope="module")
def mlflow_logger():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlflow/test_mllogs"
        yield MLFlowLogger(
            savedir=save_dir,
            experiment_name="test_experiment",
        )


@pytest.fixture
def wandb_logger():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "wandb/test_mllogs"
        yield WandBLogger(savedir=save_dir, project_name="test_project", offline_mode=True)


@pytest.fixture
def tensorboard_logger_tf():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "tf_tb/test_mllogs"
        yield TensorBoardLogger(savedir=save_dir, framework="tensorflow")


@pytest.fixture
def tensorboard_logger_torch():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "torch_tb/test_mllogs"
        yield TensorBoardLogger(savedir=save_dir, framework="pytorch")


@pytest.fixture
def prov4ml_logger():
    return Prov4MLLogger()


@pytest.fixture
def loggers_collection(
    console_logger,
    mlflow_logger,
    wandb_logger,
    tensorboard_logger_torch,
    prov4ml_logger,
):
    return LoggersCollection(
        [
            console_logger,
            mlflow_logger,
            wandb_logger,
            tensorboard_logger_torch,
            prov4ml_logger,
        ]
    )


@pytest.fixture
def lightning_mock_loggers():
    """Setup a generic PyTorchLightningLogger with mock loggers for testing."""
    itwinai_logger_mock = MagicMock(spec=Logger)
    # Setting default attributes as per class definition
    itwinai_logger_mock.is_initialized = False
    itwinai_logger_mock.worker_rank = 0
    lightning_logger = PyTorchLightningLogger(itwinai_logger=itwinai_logger_mock)

    return itwinai_logger_mock, lightning_logger
