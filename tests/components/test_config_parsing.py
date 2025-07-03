# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# # --------------------------------------------------------------------------------------

import sys
from unittest.mock import patch

import pytest
import yaml

from itwinai.cli import exec_pipeline


@pytest.fixture
def temp_yaml_files(tmp_path):
    """Fixture to create temporary YAML files with configurations defined in conftest.py.
    Returns a dictionary with their respective file paths.
    """
    yaml_data = {
        "my-list-pipeline": pytest.PIPE_LIST_YAML,
        "my-dict-pipeline": pytest.PIPE_DICT_YAML,
        "some.field.my-nested-pipeline": pytest.NESTED_PIPELINE,
        "my-interpolation-pipeline": pytest.INTERPOLATED_VALUES_PIPELINE,
    }

    file_paths = {}

    for name, content in yaml_data.items():
        file_path = tmp_path / f"{name.lower()}.yaml"
        with file_path.open("w") as f:
            yaml.dump(yaml.safe_load(content), f)
        file_paths[name] = file_path

    return file_paths


@pytest.mark.parametrize(
    "pipe_key",
    [
        "my-list-pipeline",
        "my-dict-pipeline",
        "some.field.my-nested-pipeline",
        "my-interpolation-pipeline",
    ],
)
def test_instantiate_and_execute_pipeline(pipe_key, temp_yaml_files, monkeypatch):
    """Test that exec_pipeline correctly instantiates and executes all components of
    the pipeline."""
    with (
        patch(
            "itwinai.tests.dummy_components.FakePreproc.__init__", return_value=None
        ) as mock_preproc_init,
        patch(
            "itwinai.tests.dummy_components.FakePreproc.execute", return_value=None
        ) as mock_preproc_exec,
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.__init__", return_value=None
        ) as mock_trainer_init,
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.execute", return_value=None
        ) as mock_trainer_exec,
    ):
        path = temp_yaml_files[pipe_key]
        args = [
            "itwinai exec_pipeline", # This doesn't actually do anything
            f"--config-dir={path.parent}",
            f"--config-path={path.parent}",
            f"--config-name={path.stem}",
            f"+pipe_key={pipe_key}",
        ]
        monkeypatch.setattr(sys, "argv", args)

        exec_pipeline()

        mock_preproc_init.assert_called_once_with(max_items=33, name="my-preproc")
        mock_preproc_exec.assert_called_once()

        mock_trainer_init.assert_called_once_with(lr=0.001, batch_size=32, name="my-trainer")
        mock_trainer_exec.assert_called_once()

def test_step_selection_list_pipeline(temp_yaml_files, monkeypatch):
    with (
        patch(
            "itwinai.tests.dummy_components.FakePreproc.__init__", return_value=None
        ) as mock_preproc_init,
        patch(
            "itwinai.tests.dummy_components.FakePreproc.execute", return_value=None
        ) as mock_preproc_exec,
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.__init__", return_value=None
        ) as mock_trainer_init,
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.execute", return_value=None
        ) as mock_trainer_exec,
    ):
        pipe_key="my-list-pipeline"
        path = temp_yaml_files[pipe_key]
        args = [
            "itwinai exec_pipeline", # This doesn't actually do anything
            f"--config-dir={path.parent}",
            f"--config-path={path.parent}",
            f"--config-name={path.stem}",
            f"+pipe_key={pipe_key}",
            "+pipe_steps=[1]",
        ]
        monkeypatch.setattr(sys, "argv", args)

        exec_pipeline()

        mock_trainer_init.assert_called_once_with(lr=0.001, batch_size=32, name="my-trainer")
        mock_trainer_exec.assert_called_once()

        mock_preproc_init.assert_not_called()
        mock_preproc_exec.assert_not_called()

def test_step_selection_dict_pipeline(temp_yaml_files, monkeypatch):
    with (
        patch(
            "itwinai.tests.dummy_components.FakePreproc.__init__", return_value=None
        ) as mock_preproc_init,
        patch(
            "itwinai.tests.dummy_components.FakePreproc.execute", return_value=None
        ) as mock_preproc_exec,
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.__init__", return_value=None
        ) as mock_trainer_init,
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.execute", return_value=None
        ) as mock_trainer_exec,
    ):
        pipe_key="my-dict-pipeline"
        path = temp_yaml_files[pipe_key]
        args = [
            "itwinai exec_pipeline", # This doesn't actually do anything
            f"--config-dir={path.parent}",
            f"--config-path={path.parent}",
            f"--config-name={path.stem}",
            f"+pipe_key={pipe_key}",
            "+pipe_steps=[train-step]",
        ]
        monkeypatch.setattr(sys, "argv", args)

        exec_pipeline()

        mock_trainer_init.assert_called_once_with(lr=0.001, batch_size=32, name="my-trainer")
        mock_trainer_exec.assert_called_once()

        mock_preproc_init.assert_not_called()
        mock_preproc_exec.assert_not_called()


def test_dynamic_override_interpolation_list_pipeline(temp_yaml_files, monkeypatch):
    """Test that exec_pipeline can correctly set keys via cli override and instantiate
    and execute the steps with the updated arguments. This tests the list pipeline with
    interpolation keys."""
    with (
        patch(
            "itwinai.tests.dummy_components.FakeTrainer.__init__", return_value=None
        ) as mock_trainer_init,
    ):
        pipe_key = 'my-interpolation-pipeline'
        path = temp_yaml_files[pipe_key]

        args = [
            "itwinai exec_pipeline", # This doesn't actually do anything
            f"--config-path={path.parent}",
            f"--config-dir={path.parent}",
            f"--config-name={path.stem}",
            f"+pipe_key={pipe_key}",
            "lr=0.005",
            "my-interpolation-pipeline.steps.1.batch_size=64",
        ]
        monkeypatch.setattr(sys, "argv", args)
        exec_pipeline()

        mock_trainer_init.assert_called_once_with(lr=0.005, batch_size=64, name="my-trainer")


def test_dynamic_override_cli_dict_pipeline(temp_yaml_files, monkeypatch):
    """Test that exec_pipeline can correctly set keys via cli override and instantiate
    and execute the steps with the updated arguments. This tests the dictionary pipeline."""
    with (
        patch(
            "itwinai.tests.dummy_components.FakePreproc.__init__", return_value=None
        ) as mock_preproc_init,
    ):
        pipe_key = 'my-dict-pipeline'
        path = temp_yaml_files[pipe_key]
        args = [
            "itwinai exec_pipeline", # This doesn't actually do anything
            f"--config-path={path.parent}",
            f"--config-dir={path.parent}",
            f"--config-name={path.stem}",
            f"+pipe_key={pipe_key}",
            "my-dict-pipeline.steps.preproc-step.name=new-preproc",
        ]
        monkeypatch.setattr(sys, "argv", args)
        exec_pipeline()

        mock_preproc_init.assert_called_once_with(max_items=33, name="new-preproc")


def test_invalid_step_selection(temp_yaml_files, monkeypatch):
    """Test that a call to exec_pipeline throws a 'ConfigKeyError' error if it is called
    with an invalid step to select."""
    args = [
        "itwinai exec_pipeline", # This doesn't actually do anything
        f"--config-path={temp_yaml_files['my-dict-pipeline']}",
        "pipe_key=my-dict-pipeline",
        "+pipe_steps=[non-existent-step]",
    ]
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(SystemExit) as sys_exit:
        exec_pipeline()
    assert sys_exit.value.code == 1, "Expected this to fail"


def test_invalid_pipeline_key(temp_yaml_files, monkeypatch):
    """Test that a call to exec_pipeline throws a 'MissingMandatoryValue' error if it is called
    with an invalid pipeline key to select"""
    args = [
        "itwinai exec_pipeline", # This doesn't actually do anything
        f"--config-path={temp_yaml_files['my-dict-pipeline']}",
        "pipe_key=non-existent-pipeline",
    ]
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(SystemExit) as sys_exit:
        exec_pipeline()
    assert sys_exit.value.code == 1, "Expected this to fail"
