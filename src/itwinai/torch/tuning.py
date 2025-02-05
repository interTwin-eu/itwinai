# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Dict

from ray.tune.schedulers import (
    AsyncHyperBandScheduler,
    HyperBandForBOHB,
    HyperBandScheduler,
    PopulationBasedTraining,
)
from ray.tune.schedulers.pb2 import PB2  # Population Based Bandits
from ray.tune.search.ax import AxSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.zoopt import ZOOptSearch


def get_raytune_search_alg(
    tune_config: Dict,
) -> (
    TuneBOHB
    | BayesOptSearch
    | HyperOptSearch
    | AxSearch
    | HEBOSearch
    | NevergradSearch
    | OptunaSearch
    | ZOOptSearch
    | None
):
    """Get the appropriate Ray Tune search algorithm based on the provided configuration.

    Args:
        tune_config (Dict): Configuration dictionary specifying the search algorithm,
            metric, mode, and, depending on the search algorithm, other parameters.
        seeds (bool, optional): Whether to use a fixed seed for reproducibility for some
            search algorithms that take a seed. Defaults to False.

    Returns:
        An instance of the chosen Ray Tune search algorithm or None if no search algorithm is
        used or if the search algorithm does not match any of the supported options.
    """
    scheduler_name = tune_config.get("scheduler", {}).get("name", "")

    match scheduler_name.lower():
        case "pbt" | "pb2":
            print(
                f"INFO: Using scheduler {scheduler_name} "
                "is not compatible with Ray Tune search algorithms."
            )
            print(f"Using the Ray Tune {scheduler_name} scheduler without search algorithm")
            return None

        case "bohb":
            print(
                "INFO: Using TuneBOHB search algorithm since it is required for BOHB "
                "scheduler."
            )
            return TuneBOHB()

    search_alg = tune_config.pop("search_alg", {})
    search_alg_name = search_alg.pop("name", "")

    try:
        match search_alg_name.lower():
            case "ax":
                return AxSearch()
            case "bayesopt":
                return BayesOptSearch(**search_alg)
            case "hyperopt":
                return HyperOptSearch(**search_alg)
            case "bohb":
                return TuneBOHB(**search_alg)
            case "hepo":
                return HEBOSearch(**search_alg)
            case "nevergrad":
                return NevergradSearch(**search_alg)
            case "optuna":
                return OptunaSearch(**search_alg)
            case "zoo":
                return ZOOptSearch(**search_alg)
            case _:
                print(
                    "INFO: No search algorithm detected. Using Ray Tune BasicVariantGenerator."
                )
                return None
    except AttributeError as e:
        print(
            "Invalid search algorithm configuration passed. Please make sure that the search "
            "algorithm you are using has the correct attributes. You can read more about the "
            "different search algorithms supported by Ray Tune at "
            "https://docs.ray.io/en/latest/tune/api/suggestion.html. "
        )
        print(e)


def get_raytune_scheduler(
    tune_config: Dict,
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

    scheduler = tune_config.pop("scheduler", {})
    scheduler_name = scheduler.pop("name", "")

    try:
        match scheduler_name.lower():
            case "asha":
                return AsyncHyperBandScheduler(**scheduler)
            case "hyperband":
                return HyperBandScheduler(**scheduler)
            case "bohb":
                return HyperBandForBOHB(**scheduler)
            case "pbt":
                return PopulationBasedTraining(**scheduler)
            case "pb2":
                return PB2(**scheduler)
            case _:
                print(
                    "INFO: No search algorithm detected. Using default Ray Tune FIFOScheduler."
                )
                return None
    except AttributeError as e:
        print(
            "Invalid scheduler configuration passed. Please make sure that the scheduler "
            "you are using has the correct attributes. You can read more about the "
            "different schedulers supported by Ray Tune at "
            "https://docs.ray.io/en/latest/tune/api/schedulers.html."
        )
        print(e)
