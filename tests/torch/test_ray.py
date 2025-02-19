import ray.train
import ray.tune


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

ray_tune_config = {
    "num_samples": 1,
    "scheduler": {
        "name": "asha",
        "max_t": 5,
        "grace_period": 2,
        "reduction_factor": 6,
        "brackets": 1,
    },
}

ray_run_config = {
    "storage_path": "ray_checkpoints",
    "name": "MNIST-HPO-Experiment",
}

ray_search_space = {
    "batch_size": {"type": "choice", "categories": [2, 4]},
    "optim_lr": {"type": "uniform", "lower": 1e-5, "upper": 1e-3},
}


def test_tune_config():
    """Test tune config parser"""
    config = tune_config(ray_tune_config)
    assert isinstance(config, ray.tune.TuneConfig)
    assert config.num_samples == 1
    assert config.metric == "loss"
    assert config.mode == "min"

    config = tune_config(None)
    assert config is None


def test_scaling_config():
    """Test scaling config parser"""
    config = scaling_config(ray_scaling_config)
    assert isinstance(config, ray.train.ScalingConfig)
    assert config.num_workers == 2
    assert config.use_gpu is True
    assert config.resources_per_worker["CPU"] == 4
    assert config.resources_per_worker["GPU"] == 1

    config = scaling_config(None)
    assert config is None


def test_run_config():
    """Test run config parser"""
    config = run_config(ray_run_config)
    assert isinstance(config, ray.train.RunConfig)
    assert str(config.storage_path).endswith("ray_checkpoints")
    assert config.name == "MNIST-HPO-Experiment"

    config = run_config(None)
    assert config is None


def test_search_space():
    """Test search space config parser"""
    from ray.tune.search.sample import Categorical, Float

    search_params = search_space(ray_search_space)
    assert isinstance(search_params, dict)
    assert "batch_size" in search_params
    assert "optim_lr" in search_params

    assert isinstance(search_params["batch_size"], Categorical)
    assert search_params["batch_size"].categories == [2, 4]

    assert isinstance(search_params["optim_lr"], Float)
    assert search_params["optim_lr"].lower == 1e-5
    assert search_params["optim_lr"].upper == 1e-3

    search_params = search_space(None)
    assert isinstance(search_params, dict)
    assert search_params == {}
