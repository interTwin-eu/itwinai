import yaml
import pytest

from itwinai.pipeline import Pipeline
from itwinai.parser import ConfigParser
from itwinai.tests import (
    FakeGetterExec, FakeSplitterExec, FakeTrainerExec, FakeSaverExec
)


def test_slice_into_sub_pipelines():
    """Test slicing the pipeline to obtain
    sub-pipelines as Pipeline objects.
    """
    p = Pipeline(['step1', 'step2', 'step3'])
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


def test_serialization_pipe_list():
    """Test dict serialization of pipeline
    defined as list of BaseComponent objects.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)
    parser = ConfigParser(config=config)
    pipe = parser.parse_pipeline(
        pipeline_nested_key="my-list-pipeline"
    )

    dict_pipe = pipe.to_dict()
    del dict_pipe['init_args']['name']
    dict_pipe = {"my-list-pipeline": dict_pipe}
    assert dict_pipe == config


def test_serialization_pipe_dict():
    """Test dict serialization of pipeline
    defined as dict of BaseComponent objects.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    pipe = parser.parse_pipeline(
        pipeline_nested_key="my-dict-pipeline"
    )

    dict_pipe = pipe.to_dict()
    del dict_pipe['init_args']['name']
    dict_pipe = {"my-dict-pipeline": dict_pipe}
    assert dict_pipe == config


def test_arguments_mismatch():
    """Test mismatch of arguments passed among components in a pipeline."""
    pipeline = Pipeline([
        FakeGetterExec(data_uri='http://...'),
        FakeSplitterExec(train_prop=.7),
        FakeTrainerExec(lr=1e-3, batch_size=32),
        # Adapter(policy=[f"{Adapter.INPUT_PREFIX}-1"]),
        FakeSaverExec(save_path="my_model.pth")
    ])
    with pytest.raises(TypeError) as exc_info:
        _ = pipeline.execute()
    assert "received too many input arguments" in str(exc_info.value)

    pipeline = Pipeline([
        FakeGetterExec(data_uri='http://...'),
        FakeTrainerExec(lr=1e-3, batch_size=32),
    ])
    with pytest.raises(TypeError) as exc_info:
        _ = pipeline.execute()
    assert "received too few input arguments" in str(exc_info.value)
