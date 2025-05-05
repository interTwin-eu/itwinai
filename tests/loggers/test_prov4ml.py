import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from itwinai.loggers import Prov4MLLogger


@pytest.fixture
def logger_instance():
    return Prov4MLLogger()


@pytest.fixture
def mlflow_run():
    with tempfile.TemporaryDirectory() as mlflow_temp_dir:
        mlflow.set_tracking_uri(Path(mlflow_temp_dir).resolve().as_uri())
        experiment_id = mlflow.create_experiment("temporary_experiment")
        mlflow.set_experiment(experiment_id=experiment_id)
        # nested=True is needed
        yield mlflow.start_run(nested=True)
        mlflow.end_run()



def test_create_destroy_logger_context(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    with patch("prov4ml.start_run") as start_run, patch("prov4ml.end_run") as end_run:
        logger_instance.create_logger_context(rank=1)
        logger_instance.destroy_logger_context()
        end_run.assert_called_once()
        start_run.assert_called_once()


def test_create_destroy_logger_context_should_not_log(logger_instance):
    logger_instance.should_log = MagicMock(return_value=False)
    with patch("prov4ml.start_run") as start_run, patch("prov4ml.end_run") as end_run:
        logger_instance.create_logger_context(rank=1)
        logger_instance.destroy_logger_context()
        end_run.assert_not_called()
        start_run.assert_not_called()


def test_logger_context_start_logging(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    with patch("prov4ml.start_run") as start_run, patch("prov4ml.end_run") as end_run:
        with logger_instance.start_logging(rank=1):
            pass
        start_run.assert_called_once()
        end_run.assert_called_once()


def test_create_logger_context_should_log(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)
    with patch("prov4ml.log_metric") as log_metric:
        logger_instance.log(123, "number", kind="metric")
    log_metric.assert_called_once()


def test_create_logger_context_should_not_log(logger_instance):
    logger_instance.should_log = MagicMock(return_value=False)
    logger_instance.create_logger_context(rank=1)
    with patch("prov4ml.log_metric") as log_metric:
        logger_instance.log(123, "number", kind="metric")
    log_metric.assert_not_called()


def test_save_hyperparameters(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)
    params = {"learning_rate": 0.01, "batch_size": 32}
    with patch("prov4ml.log_param") as log_param:
        logger_instance.save_hyperparameters(params)
        log_param.assert_any_call(key="learning_rate", value=0.01)
        log_param.assert_any_call(key="batch_size", value=32)


def test_log_metric(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    with patch("prov4ml.log_metric") as log_metric:
        logger_instance.log(item=0.95, identifier="accuracy", kind="metric", step=1)

        log_metric.assert_called_once_with(
            key="accuracy", value=0.95, step=1, context="training"
        )


def test_log_flops_per_batch(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    model_mock, batch_mock = MagicMock(), MagicMock()
    with patch("prov4ml.log_flops_per_batch") as log_flops_pb:
        logger_instance.log(
            item=(model_mock, batch_mock), identifier="my_flops_pb", kind="flops_pb", step=1
        )

        log_flops_pb.assert_called_once_with(
            model=model_mock, batch=batch_mock, label="my_flops_pb", step=1, context="training"
        )


def test_log_flops_per_epoch(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    model_mock, dataset_mock = MagicMock(), MagicMock()
    with patch("prov4ml.log_flops_per_epoch") as log_flops_pe:
        logger_instance.log(
            item=(model_mock, dataset_mock), identifier="my_flops_pe", kind="flops_pe", step=1
        )

        log_flops_pe.assert_called_once_with(
            model=model_mock,
            dataset=dataset_mock,
            label="my_flops_pe",
            step=1,
            context="training",
        )


def test_log_system(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    with patch("prov4ml.log_system_metrics") as log_system:
        logger_instance.log(item=None, identifier=None, kind="system", step=1)

        log_system.assert_called_once_with(context="training", step=1)


def test_log_carbon(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    with patch("prov4ml.log_carbon_metrics") as log_carbon:
        logger_instance.log(item=None, identifier=None, kind="carbon", step=1)

        log_carbon.assert_called_once_with(context="training", step=1)


def test_log_execution_time(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    with patch("prov4ml.log_current_execution_time") as log_exec_time:
        logger_instance.log(
            item=None, identifier="execution_time", kind="execution_time", step=1
        )

        log_exec_time.assert_called_once_with(
            label="execution_time", context="training", step=1
        )


def test_log_model(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    model_mock = MagicMock()
    with patch("prov4ml.save_model_version") as log_model:
        logger_instance.log(item=model_mock, identifier="model_v1", kind="model", step=1)

        log_model.assert_called_once_with(
            model=model_mock, model_name="model_v1", context="training", step=1
        )


def test_log_best_model(logger_instance):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    model_mock = MagicMock()
    with patch("prov4ml.log_model") as log_best_model:
        logger_instance.log(
            item=model_mock, identifier="best_model_v1", kind="best_model", step=1
        )

        log_best_model.assert_called_once_with(
            model=model_mock,
            model_name="best_model_v1",
            log_model_info=True,
            log_as_artifact=True,
        )


def test_log_prov_documents(logger_instance, mlflow_run):
    logger_instance.should_log = MagicMock(return_value=True)
    logger_instance.create_logger_context(rank=1)

    with patch("prov4ml.log_provenance_documents") as log_prov_documents:
        log_prov_documents.return_value = ["doc1", "doc2"]

        with patch("mlflow.log_artifact") as mlflow_log_artifact:
            logger_instance.log(item=None, identifier=None, kind="prov_documents", step=1)

            log_prov_documents.assert_called_once_with(create_graph=True, create_svg=True)
            mlflow_log_artifact.assert_any_call("doc1")
            mlflow_log_artifact.assert_any_call("doc2")