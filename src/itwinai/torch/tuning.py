from ray.tune.schedulers import (
    AsyncHyperBandScheduler,
    HyperBandForBOHB,
    HyperBandScheduler,
    PopulationBasedTraining,
)
from ray.tune.schedulers.pb2 import PB2  # Population Based Bandits
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hyperopt import HyperOptSearch


def get_raytune_search_alg(
    tune_config, seeds=False
) -> TuneBOHB | BayesOptSearch | HyperOptSearch | None:
    """Get the appropriate Ray Tune search algorithm based on the provided configuration.

    Args:
        tune_config (Dict): Configuration dictionary specifying the search algorithm,
            metric, mode, and, depending on the search algorithm, other parameters.
        seeds (bool, optional): Whether to use a fixed seed for reproducibility for some
            search algorithms that take a seed. Defaults to False.

    Returns:
        An instance of the chosen Ray Tune search algorithm or None if no search algorithm is
            used or if the search algorithm does not match any of the supported options.

    Notes:
        - `TuneBOHB` is automatically chosen for BOHB scheduling.
    """
    scheduler = tune_config.get("scheduler", {}).get("name")

    search_alg = tune_config.get("search_alg", {}).get("name")

    if (scheduler == "pbt") or (scheduler == "pb2"):
        if search_alg is None:
            return None
        else:
            print(
                "INFO: Using schedule '{}' \
                    is not compatible with Ray Tune search algorithms.".format(scheduler)
            )
            print(
                "INFO: Using the Ray Tune '{}' scheduler without search algorithm".format(
                    scheduler
                )
            )

    if (scheduler == "bohb") or (scheduler == "BOHB"):
        print("INFO: Using TuneBOHB search algorithm since it is required for BOHB shedule")
        if seeds:
            seed = 1234
        else:
            seed = None
        return TuneBOHB(
            seed=seed,
        )

    # requires pip install bayesian-optimization
    if search_alg == "bayes":
        print("INFO: Using BayesOptSearch")
        return BayesOptSearch(
            random_search_steps=tune_config["search_alg"]["n_random_steps"],
        )

    # requires pip install hyperopt
    if search_alg == "hyperopt":
        print("INFO: Using HyperOptSearch")
        return HyperOptSearch(
            n_initial_points=tune_config["search_alg"]["n_random_steps"],
            # points_to_evaluate=,
        )

    print("INFO: Not using any Ray Tune search algorithm")
    return None


def get_raytune_schedule(
    tune_config,
) -> (
    AsyncHyperBandScheduler
    | HyperBandScheduler
    | HyperBandForBOHB
    | PopulationBasedTraining
    | PB2
    | None
):
    """Get the appropriate Ray Tune scheduler based on the provided configuration.

    Args:
        tune_config (Dict): Configuration dictionary specifying the scheduler type,
            metric, mode, and, depending on the scheduler, other parameters.
    Returns:
        An instance of the chosen Ray Tune scheduler or None if no scheduler is used
            or if the scheduler does not match any of the supported options.
    """
    scheduler = tune_config.get("scheduler", {}).get("name")

    if scheduler == "asha":
        return AsyncHyperBandScheduler(
            time_attr="training_iteration",
            max_t=tune_config["scheduler"]["max_t"],
            grace_period=tune_config["scheduler"]["grace_period"],
            reduction_factor=tune_config["scheduler"]["reduction_factor"],
            brackets=tune_config["scheduler"]["brackets"],
        )
    elif scheduler == "hyperband":
        return HyperBandScheduler(
            time_attr="training_iteration",
            max_t=tune_config["scheduler"]["max_t"],
            reduction_factor=tune_config["scheduler"]["reduction_factor"],
        )
    # requires pip install hpbandster ConfigSpace
    elif (scheduler == "bohb") or (scheduler == "BOHB"):
        return HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=tune_config["scheduler"]["max_t"],
            reduction_factor=tune_config["scheduler"]["reduction_factor"],
        )
    elif (scheduler == "pbt") or (scheduler == "PBT"):
        return PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=tune_config["scheduler"]["perturbation_interval"],
            hyperparam_mutations=tune_config["scheduler"]["hyperparam_mutations"],
            log_config=True,
        )
    # requires pip install GPy sklearn
    elif (scheduler == "pb2") or (scheduler == "PB2"):
        return PB2(
            time_attr="training_iteration",
            perturbation_interval=tune_config["scheduler"]["perturbation_interval"],
            hyperparam_bounds=tune_config["scheduler"]["hyperparam_bounds"],
            log_config=True,
        )
    else:
        print("INFO: Not using any Ray Tune trial scheduler.")
        return None
