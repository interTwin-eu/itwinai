from ray.tune.schedulers import (AsyncHyperBandScheduler, HyperBandForBOHB,
                                 HyperBandScheduler, PopulationBasedTraining)
from ray.tune.schedulers.pb2 import PB2  # Population Based Bandits
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hyperopt import HyperOptSearch


def get_raytune_search_alg(tune_config, seeds=False):
    if "scheduler" in tune_config:
        scheduler = tune_config["scheduler"]["name"]
    else:
        scheduler = None
    search_alg = tune_config["search_alg"]["name"]
    metric = tune_config["metric"]
    mode = tune_config["mode"]

    if (scheduler == "pbt") or (scheduler == "pb2"):
        if search_alg is not None:
            print(
                "INFO: Using schedule '{}' \
                    is not compatible with Ray Tune search algorithms.".format(scheduler)
            )
            print(
                "INFO: Using the Ray Tune '{}' scheduler without search algorithm".format(
                    scheduler
                )
            )
        return None

    if (scheduler == "bohb") or (scheduler == "BOHB"):
        print("INFO: Using TuneBOHB search algorithm since it is required for BOHB shedule")
        if seeds:
            seed = 1234
        else:
            seed = None
        return TuneBOHB(
            metric=metric,
            mode=mode,
            seed=seed,
        )

    # requires pip install bayesian-optimization
    if search_alg == "bayes":
        print("INFO: Using BayesOptSearch")
        return BayesOptSearch(
            metric=metric,
            mode=mode,
            random_search_steps=tune_config["search_alg"]["n_random_steps"],
        )

    # requires pip install hyperopt
    if search_alg == "hyperopt":
        print("INFO: Using HyperOptSearch")
        return HyperOptSearch(
            metric=metric,
            mode=mode,
            n_initial_points=tune_config["search_alg"]["n_random_steps"],
            # points_to_evaluate=,
        )
    else:
        print("INFO: Not using any Ray Tune search algorithm")
        return None


def get_raytune_schedule(tune_config):
    scheduler = tune_config["scheduler"]["name"]
    metric = tune_config["metric"]
    mode = tune_config["mode"]

    if scheduler == "asha":
        return AsyncHyperBandScheduler(
            metric=metric,
            mode=mode,
            time_attr="training_iteration",
            max_t=tune_config["scheduler"]["max_t"],
            grace_period=tune_config["scheduler"]["grace_period"],
            reduction_factor=tune_config["scheduler"]["reduction_factor"],
            brackets=tune_config["scheduler"]["brackets"],
        )
    elif scheduler == "hyperband":
        return HyperBandScheduler(
            metric=metric,
            mode=mode,
            time_attr="training_iteration",
            max_t=tune_config["scheduler"]["max_t"],
            reduction_factor=tune_config["scheduler"]["reduction_factor"],
        )
    # requires pip install hpbandster ConfigSpace
    elif (scheduler == "bohb") or (scheduler == "BOHB"):
        return HyperBandForBOHB(
            metric=metric,
            mode=mode,
            time_attr="training_iteration",
            max_t=tune_config["scheduler"]["max_t"],
            reduction_factor=tune_config["scheduler"]["reduction_factor"],
        )
    elif (scheduler == "pbt") or (scheduler == "PBT"):
        return PopulationBasedTraining(
            metric=metric,
            mode=mode,
            time_attr="training_iteration",
            perturbation_interval=tune_config["scheduler"]["perturbation_interval"],
            hyperparam_mutations=tune_config["scheduler"]["hyperparam_mutations"],
            log_config=True,
        )
    # requires pip install GPy sklearn
    elif (scheduler == "pb2") or (scheduler == "PB2"):
        return PB2(
            metric=metric,
            mode=mode,
            time_attr="training_iteration",
            perturbation_interval=tune_config["scheduler"]["perturbation_interval"],
            hyperparam_bounds=tune_config["scheduler"]["hyperparam_bounds"],
            log_config=True,
        )
    else:
        print("INFO: Not using any Ray Tune trial scheduler.")
        return None
