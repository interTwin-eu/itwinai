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

from itwinai.components import BaseComponent
from itwinai.parser import ConfigParser
from itwinai.tests import FakePreproc, FakeSaver, FakeTrainer


def test_parse_list_pipeline():
    """Parse a pipeline from config file,
    where the pipeline is define as a list of components.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)
    parser = ConfigParser(config=config)
    pipe = parser.build_from_config(pipeline_nested_key="my-list-pipeline")

    assert isinstance(pipe.steps, list)
    for step in pipe.steps:
        assert isinstance(step, BaseComponent)


def test_parse_dict_pipeline():
    """Parse a pipeline from config file,
    where the pipeline is define as a dict of components.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    pipe = parser.build_from_config(pipeline_nested_key="my-dict-pipeline")

    assert isinstance(pipe.steps, dict)
    for step in pipe.steps.values():
        assert isinstance(step, BaseComponent)


def test_parse_non_existing_pipeline():
    """Parse a pipeline from config file, where the pipeline key is wrong."""
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    with pytest.raises(KeyError):
        _ = parser.build_from_config(pipeline_nested_key="non-existing-pipeline")


def test_parse_nested_pipeline():
    """Parse a pipeline from config file, where the pipeline key is nested."""
    config = yaml.safe_load(pytest.NESTED_PIPELINE)
    parser = ConfigParser(config=config)
    _ = parser.build_from_config(pipeline_nested_key="some.field.nst-pipeline")


def test_dynamic_override_parser_pipeline_dict():
    """Parse a pipeline from config file, and verify that dynamic override works
    in a pipeline composed of a dict of components.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)

    override_keys = {"my-dict-pipeline.init_args.steps.preproc-step.init_args.max_items": 33}
    parser = ConfigParser(config=config)
    pipe = parser.build_from_config(
        pipeline_nested_key="my-dict-pipeline", override_keys=override_keys
    )
    assert pipe.steps["preproc-step"].max_items == 33


def test_dynamic_override_parser_pipeline_list():
    """Parse a pipeline from config file, and verify that dynamic override works
    in a pipeline composed of a list of components.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)

    override_keys = {"my-list-pipeline.init_args.steps.0.init_args.max_items": 42}
    parser = ConfigParser(config=config)
    pipe = parser.build_from_config(
        pipeline_nested_key="my-list-pipeline", override_keys=override_keys
    )
    assert pipe.steps[0].max_items == 42


def test_parse_step_list_pipeline():
    """Parse a pipeline step from config file,
    where the pipeline is define as a list of components.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)
    parser = ConfigParser(config=config)
    step = parser.parse_step(step_idx=1, pipeline_nested_key="my-list-pipeline")

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakeTrainer)

    with pytest.raises(IndexError):
        _ = parser.parse_step(step_idx=12, pipeline_nested_key="my-list-pipeline")
    with pytest.raises(TypeError):
        _ = parser.parse_step(step_idx="my-step-name", pipeline_nested_key="my-list-pipeline")


def test_parse_step_dict_pipeline():
    """Parse a pipeline step from config file,
    where the pipeline is define as a dict of components.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    step = parser.parse_step(step_idx="preproc-step", pipeline_nested_key="my-dict-pipeline")

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakePreproc)

    with pytest.raises(KeyError):
        _ = parser.parse_step(step_idx="unk-step", pipeline_nested_key="my-dict-pipeline")
    with pytest.raises(KeyError):
        _ = parser.parse_step(step_idx=0, pipeline_nested_key="my-dict-pipeline")


def test_parse_step_nested_pipeline():
    """Parse a pipeline step from config file,
    where the pipeline is nested under some field.
    """
    config = yaml.safe_load(pytest.NESTED_PIPELINE)
    parser = ConfigParser(config=config)
    step = parser.parse_step(step_idx=2, pipeline_nested_key="some.field.nst-pipeline")

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakeSaver)

    with pytest.raises(KeyError):
        _ = parser.parse_step(step_idx=2, pipeline_nested_key="my-pipeline")
