import shutil
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


@pytest.fixture(scope="function")
def console_logger():
    yield ConsoleLogger(savedir="/tmp/console/test_mllogs", log_freq=1)
    shutil.rmtree("/tmp/console/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="function")
def mlflow_logger():
    yield MLFlowLogger(
        savedir="/tmp/mlflow/test_mllogs",
        experiment_name="test_experiment",
        tracking_uri="file:///tmp/mlruns",
    )
    shutil.rmtree("/tmp/mlflow/test_mllogs", ignore_errors=True)
    shutil.rmtree("/tmp/mlruns", ignore_errors=True)


@pytest.fixture(scope="function")
def wandb_logger():
    yield WandBLogger(savedir="/tmp/wandb/test_mllogs", project_name="test_project")
    shutil.rmtree("/tmp/wandb/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="function")
def tensorboard_logger_tf():
    yield TensorBoardLogger(savedir="/tmp/tf_tb/test_mllogs", framework="tensorflow")
    shutil.rmtree("/tmp/tf_tb/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="function")
def tensorboard_logger_torch():
    yield TensorBoardLogger(savedir="/tmp/torch_tb/test_mllogs", framework="pytorch")
    shutil.rmtree("/tmp/torch_tb/test_mllogs", ignore_errors=True)


@pytest.fixture(scope="function")
def prov4ml_logger():
    return Prov4MLLogger()


@pytest.fixture(scope="function")
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
