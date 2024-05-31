import pytest
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

from itwinai.loggers import (
    ConsoleLogger,
    MLFlowLogger,
    WanDBLogger,
    TensorBoardLogger,
    LoggersCollection
)


@pytest.fixture(scope="module")
def console_logger():
    yield ConsoleLogger(savedir='/tmp/console/test_mllogs', log_freq=1)
    shutil.rmtree('/tmp/console/test_mllogs', ignore_errors=True)


@pytest.fixture(scope="module")
def mlflow_logger():
    yield MLFlowLogger(
        savedir='/tmp/mlflow/test_mllogs',
        experiment_name='test_experiment',
        tracking_uri='file:///tmp/mlruns'
    )
    shutil.rmtree('/tmp/mlflow/test_mllogs', ignore_errors=True)
    shutil.rmtree('/tmp/mlruns', ignore_errors=True)


@pytest.fixture(scope="module")
def wandb_logger():
    yield WanDBLogger(savedir='/tmp/wandb/test_mllogs',
                      project_name='test_project')
    shutil.rmtree('/tmp/wandb/test_mllogs', ignore_errors=True)


@pytest.fixture(scope="module")
def tensorboard_logger_tf():
    yield TensorBoardLogger(savedir='/tmp/tf_tb/test_mllogs',
                            framework='tensorflow')
    shutil.rmtree('/tmp/tf_tb/test_mllogs', ignore_errors=True)


@pytest.fixture(scope="module")
def tensorboard_logger_torch():
    yield TensorBoardLogger(savedir='/tmp/torch_tb/test_mllogs',
                            framework='pytorch')
    shutil.rmtree('/tmp/torch_tb/test_mllogs', ignore_errors=True)


@pytest.fixture
def loggers_collection(console_logger, mlflow_logger, wandb_logger):
    return LoggersCollection([console_logger, mlflow_logger, wandb_logger])


def test_console_logger_log(console_logger):
    console_logger.create_logger_context()
    with patch('builtins.print') as mocked_print:
        console_logger.log('test_value', 'test_identifier')
        mocked_print.assert_called_with(
            'ConsoleLogger: test_identifier = test_value')
    console_logger.destroy_logger_context()


def test_mlflow_logger_log(mlflow_logger):
    with patch('mlflow.log_metric') as mock_log_metric:
        mlflow_logger.create_logger_context()
        mlflow_logger.log(0.5, 'test_metric', kind='metric', step=1)
        mock_log_metric.assert_called_once_with(
            key='test_metric', value=0.5, step=1)
        mlflow_logger.destroy_logger_context()


def test_wandb_logger_log(wandb_logger):
    with patch('wandb.init') as mock_init, patch('wandb.log') as mock_log:
        mock_init.return_value = MagicMock()
        wandb_logger.create_logger_context()
        wandb_logger.log(0.5, 'test_metric', kind='metric', step=1)
        mock_log.assert_called_once_with(
            {'test_metric': 0.5}, step=1, commit=True)


def test_tensorboard_logger_log_tf(tensorboard_logger_tf):
    import tensorflow as tf

    with patch('tensorflow.summary.scalar') as mock_scalar, \
            patch('tensorflow.summary.image') as mock_image, \
            patch('tensorflow.summary.text') as mock_text:
        tensorboard_logger_tf.create_logger_context()

        # Log a scalar
        tensorboard_logger_tf.log(0.5, 'test_scalar', kind='metric', step=1)
        mock_scalar.assert_called_once_with('test_scalar', 0.5, step=1)

        # Log an image
        image = tf.zeros([10, 10, 3])
        tensorboard_logger_tf.log(image, 'test_image', kind='image', step=1)
        mock_image.assert_called_once()
        assert np.allclose(mock_image.call_args[0][1].numpy(), image.numpy())

        # Log text
        tensorboard_logger_tf.log(
            'test text', 'test_text', kind='text', step=1)
        mock_text.assert_called_once_with('test_text', 'test text', step=1)

        tensorboard_logger_tf.destroy_logger_context()


def test_tensorboard_logger_log_torch(tensorboard_logger_torch):
    import torch

    step = 1
    with patch('torch.utils.tensorboard.SummaryWriter.add_scalar') \
            as mock_scalar, \
            patch('torch.utils.tensorboard.SummaryWriter.add_image') \
            as mock_image, \
            patch('torch.utils.tensorboard.SummaryWriter.add_text') \
            as mock_text:
        tensorboard_logger_torch.create_logger_context()

        # Log a scalar
        tensorboard_logger_torch.log(
            0.5, 'test_scalar', kind='metric', step=step)
        mock_scalar.assert_called_once_with(
            'test_scalar', 0.5, global_step=step)

        # Log an image
        image = torch.zeros([3, 10, 10])
        tensorboard_logger_torch.log(
            image, 'test_image', kind='image', step=step)
        mock_image.assert_called_once_with(
            'test_image', image, global_step=step)

        # Log text
        tensorboard_logger_torch.log(
            'test text', 'test_text', kind='text', step=step)
        mock_text.assert_called_once_with(
            'test_text', 'test text', global_step=step)

        tensorboard_logger_torch.destroy_logger_context()


def test_loggers_collection_log(loggers_collection):
    with patch('builtins.print') as mocked_print, \
            patch('mlflow.log_metric') as mock_log_metric, \
            patch('wandb.init') as mock_wandb_init, \
            patch('wandb.log') as mock_wandb_log:  # , \
        # patch('shutil.copyfile') as mock_shutil_copyfile, \
        # patch('pickle.dump') as mock_pickle_dump, \
        # patch('torch.save') as mock_torch_save:

        mock_wandb_init.return_value = MagicMock()

        loggers_collection.create_logger_context()
        loggers_collection.log(0.5, 'test_metric', kind='metric', step=1)

        mocked_print.assert_called_with('ConsoleLogger: test_metric = 0.5')
        mock_log_metric.assert_called_once_with(
            key='test_metric', value=0.5, step=1)
        mock_wandb_log.assert_called_once_with(
            {'test_metric': 0.5}, step=1, commit=True)

        loggers_collection.destroy_logger_context()
