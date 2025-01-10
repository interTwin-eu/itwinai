# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import pytest
import yaml
import omegaconf
from itwinai.components import BaseComponent
from itwinai.parser import ConfigParser
from itwinai.tests import FakePreproc, FakeSaver, FakeTrainer


@pytest.fixture
def temp_yaml_files(tmp_path):
    """Fixture to create temporary YAML files with configurations defined in conftest.py.
    Returns a dictionary with their respective file paths.
    """
    yaml_data = {
        "PIPE_LIST_YAML": pytest.PIPE_LIST_YAML,
        "PIPE_DICT_YAML": pytest.PIPE_DICT_YAML,
        "NESTED_PIPELINE": pytest.NESTED_PIPELINE,
        "INTERPOLATED_VALUES_PIPELINE": pytest.INTERPOLATED_VALUES_PIPELINE,
    }

    file_paths = {}

    for name, content in yaml_data.items():
        file_path = tmp_path / f"{name.lower()}.yaml"
        with file_path.open("w") as f:
            yaml.dump(yaml.safe_load(content), f)
        file_paths[name] = file_path

    return file_paths


def test_parse_pipeline(temp_yaml_files):
    """Parse a pipeline from config file,
    where the pipeline is define as a list of components.
    """
    parser = ConfigParser(config=temp_yaml_files["PIPE_LIST_YAML"])
    pipe = parser.build_from_config(pipeline_nested_key="my-list-pipeline")

    assert isinstance(pipe.steps, omegaconf.listconfig.ListConfig)
    for step in pipe.steps:
        assert isinstance(step, BaseComponent)


def test_parse_dict_pipeline(temp_yaml_files):
    """Parse a pipeline from config file,
    where the pipeline is define as a dict of components.
    """
    parser = ConfigParser(config=temp_yaml_files["PIPE_DICT_YAML"])
    pipe = parser.build_from_config(pipeline_nested_key="my-dict-pipeline")

    assert isinstance(pipe.steps, omegaconf.dictconfig.DictConfig)
    for step in pipe.steps.values():
        assert isinstance(step, BaseComponent)


def test_parse_non_existing_pipeline(temp_yaml_files):
    """Parse a pipeline from config file, where the pipeline key is wrong."""

    parser = ConfigParser(config=temp_yaml_files["PIPE_DICT_YAML"])
    with pytest.raises(ValueError):
        _ = parser.build_from_config(pipeline_nested_key="non-existing-pipeline")


def test_parse_nested_pipeline(temp_yaml_files):
    """Parse a pipeline from config file, where the pipeline key is nested."""
    parser = ConfigParser(config=temp_yaml_files["NESTED_PIPELINE"])
    pipe = parser.build_from_config(pipeline_nested_key="some.field.my-nested-pipeline")

    assert isinstance(pipe.steps, omegaconf.listconfig.ListConfig)
    for step in pipe.steps:
        assert isinstance(step, BaseComponent)


def test_parse_interpolation_pipeline(temp_yaml_files):
    """Parse a pipeline from config file, where the pipeline is define as a list of components
    and some values have to be resolved through variable interpolation.
    """
    parser = ConfigParser(config=temp_yaml_files["INTERPOLATED_VALUES_PIPELINE"])
    pipe = parser.build_from_config(pipeline_nested_key="my-interpolation-pipeline")

    assert pipe.steps[0].max_items == 33
    assert pipe.steps[1].name == "my-trainer"


def test_dynamic_override_parser_pipeline_dict(temp_yaml_files):
    """Parse a pipeline from config file, and verify that dynamic override works
    in a pipeline composed of a dict of components.
    """
    override_keys = {"my-dict-pipeline.steps.preproc-step.max_items": "100"}

    parser = ConfigParser(config=temp_yaml_files["PIPE_DICT_YAML"])
    pipe = parser.build_from_config(
        pipeline_nested_key="my-dict-pipeline", override_keys=override_keys
    )
    assert pipe.steps["preproc-step"].max_items == 100


def test_dynamic_override_parser_pipeline_list(temp_yaml_files):
    """Parse a pipeline from config file, and verify that dynamic override works
    in a pipeline composed of a list of components.
    """
    override_keys = {"my-list-pipeline.steps.0.max_items": "100"}

    parser = ConfigParser(config=temp_yaml_files["PIPE_LIST_YAML"])
    pipe = parser.build_from_config(
        pipeline_nested_key="my-list-pipeline", override_keys=override_keys
    )
    assert pipe.steps[0].max_items == 100


def test_parse_step_list_pipeline(temp_yaml_files):
    """Parse a pipeline step from config file,
    where the pipeline is define as a list of components.
    """
    parser = ConfigParser(config=temp_yaml_files["PIPE_LIST_YAML"])
    step = parser.build_from_config(steps=[1], pipeline_nested_key="my-list-pipeline")

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakeTrainer)

    with pytest.raises(IndexError):
        _ = parser.build_from_config(steps=[12], pipeline_nested_key="my-list-pipeline")
    with pytest.raises(omegaconf.errors.KeyValidationError):
        _ = parser.build_from_config(
            steps=["my-step-name"],
            pipeline_nested_key="my-list-pipeline",
        )


def test_parse_step_dict_pipeline(temp_yaml_files):
    """Parse a pipeline step from config file,
    where the pipeline is define as a dict of components.
    """
    parser = ConfigParser(config=temp_yaml_files["PIPE_DICT_YAML"])
    step = parser.build_from_config(
        steps=["preproc-step"],
        pipeline_nested_key="my-dict-pipeline",
    )

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakePreproc)

    with pytest.raises(KeyError):
        _ = parser.build_from_config(
            steps=["unk-step"], pipeline_nested_key="my-dict-pipeline"
        )
    with pytest.raises(KeyError):
        _ = parser.build_from_config(steps=[1], pipeline_nested_key="my-dict-pipeline")


def test_parse_step_nested_pipeline(temp_yaml_files):
    """Parse a pipeline step from config file,
    where the pipeline is nested under some field.
    """
    parser = ConfigParser(config=temp_yaml_files["NESTED_PIPELINE"])
    step = parser.build_from_config(
        steps=[2], pipeline_nested_key="some.field.my-nested-pipeline"
    )

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakeSaver)

    # with pytest.raises(KeyError):
    #     _ = parser.build_from_config(steps=[-1], pipeline_nested_key="my-pipeline")
