# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import sys
from unittest.mock import patch

import pytest

from itwinai.cli import exec_pipeline
from itwinai.pipeline import Pipeline
from itwinai.tests import (
    FakeGetterExec,
    FakeSaverExec,
    FakeSplitterExec,
    FakeTrainerExec,
)
from itwinai.tests.dummy_components import FakePreproc, FakeTrainer


def test_slice_into_sub_pipelines():
    """Test slicing the pipeline to obtain sub-pipelines as Pipeline objects."""
    p = Pipeline(["step1", "step2", "step3"])
    sub_pipe1, sub_pipe2 = p[:1], p[1:]
    assert isinstance(sub_pipe1, Pipeline)
    assert isinstance(sub_pipe2, Pipeline)
    assert len(sub_pipe1) == 1
    assert sub_pipe1[0] == "step1"
    assert len(sub_pipe2) == 2

    p = Pipeline(dict(step1="step1", step2="step2", step3="step3"))
    sub_pipe1, sub_pipe2 = p[:1], p[1:]
    assert isinstance(sub_pipe1, Pipeline)
    assert isinstance(sub_pipe2, Pipeline)
    assert len(sub_pipe1) == 1
    assert sub_pipe1["step1"] == "step1"
    assert len(sub_pipe2) == 2


@pytest.mark.parametrize(
    "pipeline",
    [
        Pipeline(
            steps=[
                FakePreproc(max_items=33, name="my-preproc"),
                FakeTrainer(lr=0.001, batch_size=64, name="my-trainer"),
            ]
        ),
        Pipeline(
            steps={
                "preproc": FakePreproc(max_items=33, name="my-preproc"),
                "trainer": FakeTrainer(lr=0.001, batch_size=64, name="my-trainer"),
            }
        ),
    ],
)
def test_serialization_to_yaml(pipeline, tmp_path, monkeypatch):
    """Test serialization of pipeline to yaml."""
    yaml_path = tmp_path / "pipe.yaml"
    pipeline.to_yaml(file_path=yaml_path, nested_key="my_pipeline")

    with pytest.raises(SystemExit):
        with (
            patch(
                "itwinai.tests.dummy_components.FakePreproc.__init__", return_value=None
            ) as mock_preproc_init,
            patch(
                "itwinai.tests.dummy_components.FakeTrainer.__init__", return_value=None
            ) as mock_trainer_init,
        ):
            args = [
                "exec_pipeline",
                f"--config-path={tmp_path}",
                "pipe_key=my_pipeline",
            ]
            monkeypatch.setattr(sys, "argv", args)

            exec_pipeline()

            mock_preproc_init.assert_called_once_with(max_items=33, name="my-preproc")
            mock_trainer_init.assert_called_once_with(
                lr=0.001, batch_size=64, name="my-trainer"
            )


def test_arguments_mismatch():
    """Test mismatch of arguments passed among components in a pipeline."""
    pipeline = Pipeline(
        [
            FakeGetterExec(data_uri="http://..."),
            FakeSplitterExec(train_prop=0.7),
            FakeTrainerExec(lr=1e-3, batch_size=32),
            # Adapter(policy=[f"{Adapter.INPUT_PREFIX}-1"]),
            FakeSaverExec(save_path="my_model.pth"),
        ]
    )
    with pytest.raises(TypeError) as exc_info:
        _ = pipeline.execute()
    assert "received too many input arguments" in str(exc_info.value)

    pipeline = Pipeline(
        [
            FakeGetterExec(data_uri="http://..."),
            FakeTrainerExec(lr=1e-3, batch_size=32),
        ]
    )
    with pytest.raises(TypeError) as exc_info:
        _ = pipeline.execute()
    assert "received too few input arguments" in str(exc_info.value)
