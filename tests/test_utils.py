"""
Tests for itwinai.utils module.
"""

from itwinai.utils import flatten_dict, SignatureInspector


def test_flatten_dict():
    """
    Test flatten dict function.
    """
    dict1 = dict(a=1, b=dict(b1=2, b2=3))

    flattened = flatten_dict(dict1)
    assert flattened.get("a") == 1
    assert flattened.get("b.b1") == 2
    assert flattened.get("b.b2") == 3
    assert len(flattened) == 3


def test_signature_inspector():
    """Test SignatureInspector class."""
    def f():
        ...

    inspector = SignatureInspector(f)
    assert not inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ()
    assert inspector.min_params_num == 0
    assert inspector.max_params_num == 0

    def f(*args):
        ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ()
    assert inspector.min_params_num == 0
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, *args):
        ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ("foo",)
    assert inspector.min_params_num == 1
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, bar=123):
        ...

    inspector = SignatureInspector(f)
    assert not inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ("foo",)
    assert inspector.min_params_num == 1
    assert inspector.max_params_num == 2

    def f(foo, *args, bar=123):
        ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert not inspector.has_kwargs
    assert inspector.required_params == ("foo",)
    assert inspector.min_params_num == 1
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(*args, **kwargs):
        ...

    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert inspector.has_kwargs
    assert inspector.required_params == ()
    assert inspector.min_params_num == 0
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, /, bar, *arg, **kwargs):
        ...
    inspector = SignatureInspector(f)
    assert inspector.has_varargs
    assert inspector.has_kwargs
    assert inspector.required_params == ("foo", "bar")
    assert inspector.min_params_num == 2
    assert inspector.max_params_num == SignatureInspector.INFTY

    def f(foo, /, bar, *, hello, **kwargs):
        ...
    inspector = SignatureInspector(f)
    assert not inspector.has_varargs
    assert inspector.has_kwargs
    assert inspector.required_params == ("foo", "bar", "hello")
    assert inspector.min_params_num == 3
    assert inspector.max_params_num == SignatureInspector.INFTY
