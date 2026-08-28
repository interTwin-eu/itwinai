# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Regression tests for the MLflow backend.

These tests deliberately avoid mocking the mlflow API: they exercise a real local
tracking backend, which is what broke when mlflow 3 started rejecting file stores.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from itwinai.loggers import MLFlowLogger

FILE_STORE_OPT_OUT = "MLFLOW_ALLOW_FILE_STORE"


def test_importing_itwinai_opts_into_file_store():
    """itwinai logs to a local file store by default, so the opt-out must be set on import."""
    assert os.environ.get(FILE_STORE_OPT_OUT) == "true"


def test_file_store_usable_without_preset_env_var(tmp_path):
    """A local tracking backend must work in a fresh process that does not preset the opt-out.

    Runs in a subprocess because itwinai has already been imported in this one, which sets
    the environment variable process-wide and would mask the regression.
    """
    script = textwrap.dedent(f"""
        from itwinai.loggers import MLFlowLogger

        logger = MLFlowLogger(savedir=r"{tmp_path}", experiment_name="regression")
        logger.create_logger_context()
        logger.log(1.0, "a_metric", kind="metric", step=0)
        logger.destroy_logger_context()
        print("OK")
    """)

    env = {k: v for k, v in os.environ.items() if k != FILE_STORE_OPT_OUT}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )

    assert result.returncode == 0, (
        f"local file store rejected with {FILE_STORE_OPT_OUT} unset:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_metric_roundtrip_through_real_backend(tmp_path):
    """Metrics must be readable back from the store, not just accepted by the fluent API."""
    logger = MLFlowLogger(savedir=tmp_path, experiment_name="roundtrip")
    logger.create_logger_context()
    run_id = logger.active_run.info.run_id
    for step, value in enumerate([0.3, 0.2, 0.1]):
        logger.log(value, "loss", kind="metric", step=step)
    logger.destroy_logger_context()

    from mlflow.tracking import MlflowClient

    history = MlflowClient(tracking_uri=logger.tracking_uri).get_metric_history(run_id, "loss")
    assert [m.value for m in history] == [0.3, 0.2, 0.1]


def test_log_torch_model_roundtrip(tmp_path):
    """``kind='model'`` must produce a loadable model.

    mlflow 3 changed ``mlflow.pytorch.log_model`` to default to the 'pt2' serialization
    format, which raises unless an ``input_example`` is supplied.
    """
    torch = pytest.importorskip("torch")
    import mlflow

    model = torch.nn.Linear(4, 2)
    logger = MLFlowLogger(savedir=tmp_path, experiment_name="models")
    logger.create_logger_context()
    logger.log(model, "my_model", kind="model")
    run_id = logger.active_run.info.run_id
    logger.destroy_logger_context()

    mlflow.set_tracking_uri(logger.tracking_uri)
    loaded = mlflow.pytorch.load_model(f"runs:/{run_id}/my_model")
    assert torch.allclose(loaded.weight, model.weight)


def test_log_model_ignores_non_mlflow_kwargs(tmp_path):
    """Extra kwargs accepted by ``log()`` must not leak into ``mlflow.pytorch.log_model``.

    Use cases pass things like ``context="training"``, which would otherwise reach
    ``torch.save`` and raise a TypeError.
    """
    torch = pytest.importorskip("torch")

    logger = MLFlowLogger(savedir=tmp_path, experiment_name="kwargs")
    logger.create_logger_context()
    logger.log(torch.nn.Linear(2, 2), "generator_epoch_0", kind="model", context="training")
    logger.destroy_logger_context()


def test_log_model_forwards_known_mlflow_kwargs(tmp_path):
    """kwargs that ``mlflow.pytorch.log_model`` does accept must still be forwarded."""
    torch = pytest.importorskip("torch")

    logger = MLFlowLogger(savedir=tmp_path, experiment_name="kwargs-fwd")
    logger.create_logger_context()
    with patch.object(logger.mlflow.pytorch, "log_model", autospec=True) as mock_log_model:
        logger.log(torch.nn.Linear(2, 2), "LSTM", kind="model", registered_model_name="LSTM")
    logger.destroy_logger_context()

    assert mock_log_model.call_args.kwargs["registered_model_name"] == "LSTM"
    assert mock_log_model.call_args.kwargs["name"] == "LSTM"


@pytest.mark.parametrize(
    "identifier,expected",
    [
        ("my_model", "my_model"),
        ("generator_epoch_1", "generator_epoch_1"),
        ("ckpts/generator", "ckpts_generator"),
        ("best_model.pth", "best_model_pth"),
    ],
)
def test_log_model_sanitizes_identifier(tmp_path, identifier, expected):
    """mlflow 3 validates model names, which the old run-relative artifact path did not.

    Path-like identifiers are valid input everywhere else in ``log()``, so they must keep
    working rather than raising MlflowException.
    """
    torch = pytest.importorskip("torch")
    import mlflow

    logger = MLFlowLogger(savedir=tmp_path, experiment_name="names")
    logger.create_logger_context()
    run_id = logger.active_run.info.run_id
    logger.log(torch.nn.Linear(2, 2), identifier, kind="model")
    logger.destroy_logger_context()

    mlflow.set_tracking_uri(logger.tracking_uri)
    assert mlflow.pytorch.load_model(f"runs:/{run_id}/{expected}") is not None


def test_tracking_uri_is_a_resolved_file_uri(tmp_path):
    """The default backend stays a local file URI under savedir."""
    logger = MLFlowLogger(savedir=tmp_path, experiment_name="uri")
    assert logger.tracking_uri.startswith("file://")
    assert Path(logger.tracking_uri.removeprefix("file://")).is_absolute()
