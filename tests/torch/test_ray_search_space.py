# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import pytest
import ray.tune
from ray.tune.search.sample import Categorical, Float, Function, Integer

from itwinai.torch.tuning import (
    search_space,
)


def test_search_space_parsing():
    """Test search space config parser for all Ray Tune types and exceptions,
    when the configuration needs to be instantiated.
    """
    search_params = search_space(
        {
            "uniform_param": {"type": "uniform", "lower": 1e-5, "upper": 1e-3},
            "quniform_param": {"type": "quniform", "lower": 1, "upper": 10, "q": 0.5},
            "loguniform_param": {"type": "loguniform", "lower": 1, "upper": 100},
            "qloguniform_param": {"type": "qloguniform", "lower": 1, "upper": 100, "q": 0.5},
            "randn_param": {"type": "randn", "mean": 0.0, "sd": 1.0},
            "qrandn_param": {"type": "qrandn", "mean": 0.0, "sd": 1.0, "q": 0.5},
            "randint_param": {"type": "randint", "lower": 1, "upper": 100},
            "qrandint_param": {"type": "qrandint", "lower": 1, "upper": 100, "q": 5},
            "lograndint_param": {"type": "lograndint", "lower": 1, "upper": 100},
            "qlograndint_param": {"type": "qlograndint", "lower": 1, "upper": 100, "q": 5},
            "choice_param": {"type": "choice", "categories": [2, 4]},
            "grid_param": {"type": "grid_search", "values": [1, 2, 3]},
        }
    )
    assert isinstance(search_params, dict)
    assert isinstance(search_params["uniform_param"], Float)
    assert search_params["uniform_param"].lower == 1e-5
    assert search_params["uniform_param"].upper == 1e-3

    assert isinstance(search_params["quniform_param"], Float)
    assert search_params["quniform_param"].lower == 1
    assert search_params["quniform_param"].upper == 10

    assert isinstance(search_params["loguniform_param"], Float)
    assert search_params["loguniform_param"].lower == 1
    assert search_params["loguniform_param"].upper == 100

    assert isinstance(search_params["qloguniform_param"], Float)
    assert search_params["qloguniform_param"].lower == 1
    assert search_params["qloguniform_param"].upper == 100

    assert isinstance(search_params["randn_param"], Float)

    assert isinstance(search_params["qrandn_param"], Float)

    assert isinstance(search_params["randint_param"], Integer)
    assert search_params["randint_param"].lower == 1
    assert search_params["randint_param"].upper == 100

    assert isinstance(search_params["qrandint_param"], Integer)
    assert search_params["qrandint_param"].lower == 1
    assert search_params["qrandint_param"].upper == 100

    assert isinstance(search_params["lograndint_param"], Integer)
    assert search_params["lograndint_param"].lower == 1
    assert search_params["lograndint_param"].upper == 100

    assert isinstance(search_params["qlograndint_param"], Integer)
    assert search_params["qlograndint_param"].lower == 1
    assert search_params["qlograndint_param"].upper == 100

    assert isinstance(search_params["choice_param"], Categorical)
    assert search_params["choice_param"].categories == [2, 4]
    assert isinstance(search_params["grid_param"], dict)

    assert search_params["grid_param"] == {"grid_search": [1, 2, 3]}

    search_params = search_space(None)
    assert isinstance(search_params, dict)
    assert search_params == {}

    with pytest.raises(ValueError):
        search_space({"invalid": "value"})

    with pytest.raises(AttributeError):
        search_space({"invalid_param": {"type": "unknown_type"}})


def test_search_space_objects():
    """Test search space config parser for all Ray Tune types and exceptions,
    when the configuration doesn't need to be instantiated.
    """
    search_params = search_space(
        {
            "uniform_param": ray.tune.uniform(lower=1e-5, upper=1e-3),
            "quniform_param": ray.tune.quniform(lower=1, upper=10, q=0.5),
            "loguniform_param": ray.tune.loguniform(lower=1, upper=100),
            "qloguniform_param": ray.tune.qloguniform(lower=1, upper=100, q=0.5),
            "randn_param": ray.tune.randn(mean=0, sd=1),
            "qrandn_param": ray.tune.qrandn(mean=0, sd=1, q=0.5),
            "randint_param": ray.tune.randint(lower=1, upper=100),
            "qrandint_param": ray.tune.qrandint(lower=1, upper=100, q=5),
            "lograndint_param": ray.tune.lograndint(lower=1, upper=100),
            "qlograndint_param": ray.tune.qlograndint(lower=1, upper=100, q=5),
            "choice_param": ray.tune.choice(categories=[2, 4]),
            "grid_param": ray.tune.grid_search(values=[1, 2, 3]),
            "func_param": ray.tune.sample_from(lambda spec: spec.config.uniform_param * 0.01),
        }
    )
    assert isinstance(search_params, dict)
    assert isinstance(search_params["uniform_param"], Float)
    assert search_params["uniform_param"].lower == 1e-5
    assert search_params["uniform_param"].upper == 1e-3

    assert isinstance(search_params["quniform_param"], Float)
    assert search_params["quniform_param"].lower == 1
    assert search_params["quniform_param"].upper == 10

    assert isinstance(search_params["loguniform_param"], Float)
    assert search_params["loguniform_param"].lower == 1
    assert search_params["loguniform_param"].upper == 100

    assert isinstance(search_params["qloguniform_param"], Float)
    assert search_params["qloguniform_param"].lower == 1
    assert search_params["qloguniform_param"].upper == 100

    assert isinstance(search_params["randn_param"], Float)

    assert isinstance(search_params["qrandn_param"], Float)

    assert isinstance(search_params["randint_param"], Integer)
    assert search_params["randint_param"].lower == 1
    assert search_params["randint_param"].upper == 100

    assert isinstance(search_params["qrandint_param"], Integer)
    assert search_params["qrandint_param"].lower == 1
    assert search_params["qrandint_param"].upper == 100

    assert isinstance(search_params["lograndint_param"], Integer)
    assert search_params["lograndint_param"].lower == 1
    assert search_params["lograndint_param"].upper == 100

    assert isinstance(search_params["qlograndint_param"], Integer)
    assert search_params["qlograndint_param"].lower == 1
    assert search_params["qlograndint_param"].upper == 100

    assert isinstance(search_params["choice_param"], Categorical)
    assert search_params["choice_param"].categories == [2, 4]
    assert isinstance(search_params["grid_param"], dict)

    assert search_params["grid_param"] == {"grid_search": [1, 2, 3]}

    assert isinstance(search_params["func_param"], Function)

    search_params = search_space(None)
    assert isinstance(search_params, dict)
    assert search_params == {}

    with pytest.raises(ValueError):
        search_space({"invalid": "value"})

    with pytest.raises(AttributeError):
        search_space({"invalid_param": {"type": "unknown_type"}})
