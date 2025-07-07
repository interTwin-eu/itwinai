# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Tests for itwinai.utils module."""

from pathlib import Path

import pytest

from itwinai.utils import SignatureInspector, flatten_dict, make_config_paths_absolute, to_uri


def test_flatten_dict():
    """Test flatten dict function."""
    dict1 = dict(a=1, b=dict(b1=2, b2=3))

    flattened = flatten_dict(dict1)
    assert flattened.get("a") == 1
    assert flattened.get("b.b1") == 2
    assert flattened.get("b.b2") == 3
    assert len(flattened) == 3


def test_signature_inspector():
    """Test SignatureInspector class."""

    def f(): ...

    inspector = SignatureInspector(f)
    assert not inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ()
    assert inspector.min_params_num == 0
    assert inspector.max_params_num == 0

    def f(*args): ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ()
    assert inspector.min_params_num == 0
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, *args): ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ("foo",)
    assert inspector.min_params_num == 1
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, bar=123): ...

    inspector = SignatureInspector(f)
    assert not inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ("foo",)
    assert inspector.min_params_num == 1
    assert inspector.max_params_num == 2

    def f(foo, *args, bar=123): ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ("foo",)
    assert inspector.min_params_num == 1
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(*args, **kwargs): ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert inspector.has_kwargs
    assert inspector.required_params == ()
    assert inspector.min_params_num == 0
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, /, bar, *arg, **kwargs): ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert inspector.has_kwargs
    assert inspector.required_params == ("foo", "bar")
    assert inspector.min_params_num == 2
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, /, bar, *, hello, **kwargs): ...

    inspector = SignatureInspector(f)
    assert not inspector.has_varargs
    assert inspector.has_kwargs
    assert inspector.required_params == ("foo", "bar", "hello")
    assert inspector.min_params_num == 3
    assert inspector.max_params_num == SignatureInspector.INFTY


@pytest.mark.parametrize(
    "args,correct_absolute_args",
    [
        (
            ["--config-path=./relative/path", "--some-other-arg"],
            [f"--config-path={Path('./relative/path').resolve()}", "--some-other-arg"],
        ),
        (
            ["--config-path", "./relative/path", "--some-other-arg"],
            ["--config-path", f"{Path('./relative/path').resolve()}", "--some-other-arg"],
        ),
        (
            ["-cp=./relative/path", "--some-other-arg"],
            [f"-cp={Path('./relative/path').resolve()}", "--some-other-arg"],
        ),
        (
            ["-cp", "./relative/path", "--some-other-arg"],
            ["-cp", f"{Path('./relative/path').resolve()}", "--some-other-arg"],
        ),
    ],
)
def test_make_config_paths_absolute(args, correct_absolute_args):
    """Test that make_config_paths_absolute correctly resolves absolute paths."""
    updated_args = make_config_paths_absolute(args)

    assert updated_args == correct_absolute_args


def test_to_uri():
    relative_path = "relative/path/to/file.txt"
    absolute_path = "/absolute/path/to/file.txt"
    s3_uri = "s3://my-bucket/data/file.txt"
    http_uri = "http://example.com/file.txt"

    assert to_uri(relative_path) == str(Path(relative_path).resolve()), "Should be absolute"
    assert to_uri(absolute_path) == absolute_path, "Should remain unchanged"
    assert to_uri(s3_uri) == s3_uri, "Should remain unchanged"
    assert to_uri(http_uri) == http_uri, "Should remain unchanged"
    assert to_uri(Path(relative_path)) == str(Path(relative_path).resolve()), (
        "Should manage Path"
    )
