# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# -------------------------------------------------------------------------------------

import shutil
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


@pytest.fixture(scope="module")
def console_logger():
    yield ConsoleLogger(savedir="/tmp/console/test_mllogs", log_freq=1)
    shutil.rmtree("/tmp/console/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="module")
def mlflow_logger():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "mlflow/test_mllogs"
        yield MLFlowLogger(
            savedir=save_dir,
            experiment_name="test_experiment",
        )


@pytest.fixture(scope="module")
def wandb_logger():
    yield WandBLogger(savedir="/tmp/wandb/test_mllogs", project_name="test_project")
    shutil.rmtree("/tmp/wandb/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="module")
def tensorboard_logger_tf():
    yield TensorBoardLogger(savedir="/tmp/tf_tb/test_mllogs", framework="tensorflow")
    shutil.rmtree("/tmp/tf_tb/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="module")
def tensorboard_logger_torch():
    yield TensorBoardLogger(savedir="/tmp/torch_tb/test_mllogs", framework="pytorch")
    shutil.rmtree("/tmp/torch_tb/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="module")
def prov4ml_logger():
    return Prov4MLLogger()


@pytest.fixture(scope="module")
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
    lightning_logger = PyTorchLightningLogger(itwinai_logger=itwinai_logger_mock)

    return itwinai_logger_mock, lightning_logger
