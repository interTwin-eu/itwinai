"""
Tests for itwinai.utils module.
"""

from itwinai.utils import flatten_dict


def test_flatten_dict():
    """
    Test flatten dict function.
    """
    dict1 = dict(
        a=1,
        b=dict(
            b1=2,
            b2=3
        )
    )

    flattened = flatten_dict(dict1)
    assert flattened.get('a') == 1
    assert flattened.get('b.b1') == 2
    assert flattened.get('b.b2') == 3
    assert len(flattened) == 3
