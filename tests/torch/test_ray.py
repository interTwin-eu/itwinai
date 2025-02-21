# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import pytest
import ray.train
import ray.tune
from ray.tune import TuneConfig
from ray.tune.schedulers import (
    AsyncHyperBandScheduler,
    HyperBandForBOHB,
    HyperBandScheduler,
    PopulationBasedTraining,
)
from ray.tune.schedulers.pb2 import PB2
from ray.tune.search.ax import AxSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.sample import Categorical, Float, Integer
from ray.tune.search.zoopt import ZOOptSearch

from itwinai.torch.ray import (
    run_config,
    scaling_config,
    search_space,
    tune_config,
)

# Sample configurations for testing
ray_scaling_config = {
    "num_workers": 2,
    "use_gpu": True,
    "resources_per_worker": {"CPU": 4, "GPU": 1},
}


ray_run_config = {
    "storage_path": "ray_checkpoints",
    "name": "MNIST-HPO-Experiment",
}


def test_tune_config():
    """Test tune config parser including search algorithms and schedulers"""
    tune_cfg = {
        "num_samples": 2,
        "metric": "accuracy",
        "mode": "max",
        "scheduler": {"name": "asha", "max_t": 5, "grace_period": 2, "reduction_factor": 6},
        "search_alg": {"name": "hyperopt"},
    }
    config = tune_config(tune_cfg)
    assert isinstance(config, TuneConfig)
    assert config.num_samples == 2
    assert config.metric == "accuracy"
    assert config.mode == "max"
    assert isinstance(config.scheduler, AsyncHyperBandScheduler)
    assert isinstance(config.search_alg, HyperOptSearch)

    schedulers = [
        ("hyperband", HyperBandScheduler),
        ("bohb", HyperBandForBOHB),
        ("pbt", PopulationBasedTraining),
        ("pb2", PB2),
    ]
    for name, sched_class in schedulers:
        tune_cfg["scheduler"] = {"name": name}
        config = tune_config(tune_cfg)
        assert isinstance(config.scheduler, sched_class)

    search_algs = [
        ("ax", AxSearch),
        ("bayesopt", BayesOptSearch),
        ("bohb", TuneBOHB),
        ("hyperopt", HyperOptSearch),
        ("hebo", HEBOSearch),
        ("nevergrad", NevergradSearch),
        ("optuna", OptunaSearch),
        ("zoo", ZOOptSearch),
    ]
    for name, search_class in search_algs:
        tune_cfg["search_alg"] = {"name": name}
        config = tune_config(tune_cfg)
        assert isinstance(config.search_alg, search_class)

    config = tune_config(None)
    assert config is None

    with pytest.raises(Exception):
        tune_config({"search_alg": {"name": "invalid"}})

    with pytest.raises(Exception):
        tune_config({"scheduler": {"name": "invalid"}})


def test_scaling_config():
    """Test scaling config parser and exception handling"""
    config = scaling_config(ray_scaling_config)
    assert isinstance(config, ray.train.ScalingConfig)
    assert config.num_workers == 2
    assert config.use_gpu is True
    assert config.resources_per_worker["CPU"] == 4
    assert config.resources_per_worker["GPU"] == 1

    config = scaling_config(None)
    assert config is None

    with pytest.raises(TypeError):
        scaling_config({"invalid_key": "value"})


def test_run_config(tmp_path):
    """Test run config parser and exception handling"""
    config = run_config(ray_run_config, tmp_path)
    assert isinstance(config, ray.train.RunConfig)
    assert str(config.storage_path).endswith("ray_checkpoints")
    assert config.name == "MNIST-HPO-Experiment"

    config = run_config(None, tmp_path)
    assert config is None

    with pytest.raises(TypeError):
        run_config({"invalid_key": "value"}, tmp_path)


def test_search_space():
    """Test search space config parser for all Ray Tune types and exceptions"""
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
