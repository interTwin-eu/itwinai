import yaml
import pytest

from itwinai.components import BaseComponent
from itwinai.parser import ConfigParser, add_replace_field
from itwinai.tests import FakeTrainer, FakePreproc, FakeSaver


def test_add_replace_field():
    conf = {}
    add_replace_field(conf, "some.key.chain", 123)
    target1 = dict(some=dict(key=dict(chain=123)))
    assert conf == target1

    add_replace_field(conf, "some.key.chain", 222)
    target2 = dict(some=dict(key=dict(chain=222)))
    assert conf == target2

    add_replace_field(conf, "some.key.field", 333)
    target3 = dict(some=dict(key=dict(chain=222, field=333)))
    assert conf == target3

    conf['some']['list'] = [1, 2, 3]
    add_replace_field(conf, "some.list.0", 3)
    target4 = dict(some=dict(
        key=dict(chain=222, field=333),
        list=[3, 2, 3]
    ))
    assert conf == target4

    add_replace_field(conf, "some.list.0.some.el", 7)
    target5 = dict(some=dict(
        key=dict(chain=222, field=333),
        list=[dict(some=dict(el=7)), 2, 3]
    ))
    assert conf == target5

    conf2 = dict(first=dict(list1=[[0, 1], [2, 3]], el=0))
    add_replace_field(conf2, "first.list1.1.0", 77)
    target6 = dict(first=dict(list1=[[0, 1], [77, 3]], el=0))
    assert conf2 == target6

    conf3 = dict(first=dict(
        list1=[[0, dict(nst=("el", dict(ciao="ciao")))], [2, 3]], el=0))
    add_replace_field(conf3, "first.list1.0.1.nst.1.ciao", "hello")
    target7 = dict(first=dict(
        list1=[[0, dict(nst=("el", dict(ciao="hello")))], [2, 3]], el=0))
    assert conf3 == target7

    add_replace_field(conf3, "first.list1.0.1.nst.1.ciao.I.am.john", True)
    target8 = dict(first=dict(
        list1=[
            [0, dict(nst=("el", dict(ciao=dict(I=dict(am=dict(john=True))))))],
            [2, 3]
        ], el=0))
    assert conf3 == target8


def test_parse_list_pipeline():
    """Parse a pipeline from config file,
    where the pipeline is define as a list of components.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)
    parser = ConfigParser(config=config)
    pipe = parser.parse_pipeline(
        pipeline_nested_key="my-list-pipeline"
    )

    assert isinstance(pipe.steps, list)
    for step in pipe.steps:
        assert isinstance(step, BaseComponent)


def test_parse_dict_pipeline():
    """Parse a pipeline from config file,
    where the pipeline is define as a dict of components.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    pipe = parser.parse_pipeline(
        pipeline_nested_key="my-dict-pipeline"
    )

    assert isinstance(pipe.steps, dict)
    for step in pipe.steps.values():
        assert isinstance(step, BaseComponent)


def test_parse_non_existing_pipeline():
    """Parse a pipeline from config file,
    where the pipeline key is wrong.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    with pytest.raises(KeyError):
        _ = parser.parse_pipeline(
            pipeline_nested_key="non-existing-pipeline"
        )


def test_parse_nested_pipeline():
    """Parse a pipeline from config file,
    where the pipeline key is nested.
    """
    config = yaml.safe_load(pytest.NESTED_PIPELINE)
    parser = ConfigParser(config=config)
    _ = parser.parse_pipeline(
        pipeline_nested_key="some.field.nst-pipeline"
    )


def test_dynamic_override_parser_pipeline_dict():
    """Parse a pipeline from config file,
    and verify that dynamic override works
    in a pipeline composed of a dict of components.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)

    override_keys = {
        "my-dict-pipeline.init_args.steps.preproc-step.init_args.max_items": 33
    }
    parser = ConfigParser(config=config, override_keys=override_keys)
    pipe = parser.parse_pipeline(
        pipeline_nested_key="my-dict-pipeline"
    )
    assert pipe.steps['preproc-step'].max_items == 33


def test_dynamic_override_parser_pipeline_list():
    """Parse a pipeline from config file,
    and verify that dynamic override works
    in a pipeline composed of a list of components.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)

    override_keys = {
        "my-list-pipeline.init_args.steps.0.init_args.max_items": 42
    }
    parser = ConfigParser(config=config, override_keys=override_keys)
    pipe = parser.parse_pipeline(
        pipeline_nested_key="my-list-pipeline"
    )
    assert pipe.steps[0].max_items == 42


def test_parse_step_list_pipeline():
    """Parse a pipeline step from config file,
    where the pipeline is define as a list of components.
    """
    config = yaml.safe_load(pytest.PIPE_LIST_YAML)
    parser = ConfigParser(config=config)
    step = parser.parse_step(
        step_idx=1,
        pipeline_nested_key="my-list-pipeline"
    )

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakeTrainer)

    with pytest.raises(IndexError):
        _ = parser.parse_step(
            step_idx=12,
            pipeline_nested_key="my-list-pipeline"
        )
    with pytest.raises(TypeError):
        _ = parser.parse_step(
            step_idx='my-step-name',
            pipeline_nested_key="my-list-pipeline"
        )


def test_parse_step_dict_pipeline():
    """Parse a pipeline step from config file,
    where the pipeline is define as a dict of components.
    """
    config = yaml.safe_load(pytest.PIPE_DICT_YAML)
    parser = ConfigParser(config=config)
    step = parser.parse_step(
        step_idx='preproc-step',
        pipeline_nested_key="my-dict-pipeline"
    )

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakePreproc)

    with pytest.raises(KeyError):
        _ = parser.parse_step(
            step_idx='unk-step',
            pipeline_nested_key="my-dict-pipeline"
        )
    with pytest.raises(KeyError):
        _ = parser.parse_step(
            step_idx=0,
            pipeline_nested_key="my-dict-pipeline"
        )


def test_parse_step_nested_pipeline():
    """Parse a pipeline step from config file,
    where the pipeline is nested under some field.
    """
    config = yaml.safe_load(pytest.NESTED_PIPELINE)
    parser = ConfigParser(config=config)
    step = parser.parse_step(
        step_idx=2,
        pipeline_nested_key="some.field.nst-pipeline"
    )

    assert isinstance(step, BaseComponent)
    assert isinstance(step, FakeSaver)

    with pytest.raises(KeyError):
        _ = parser.parse_step(
            step_idx=2,
            pipeline_nested_key="my-pipeline"
        )
