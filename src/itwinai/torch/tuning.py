# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# --------------------------------------------------------------------------------------


"""Logic to parse configuration and transform it into Ray objects."""

import logging
from typing import Dict

import ray.tune
from ray.tune.search.sample import Categorical, Float, Function, Integer

py_logger = logging.getLogger(__name__)


def search_space(config: Dict | None) -> Dict:
    if not config:
        py_logger.warning(
            "No search_space configuration detected. "
            "If you want to tune any hyperparameters, make sure to define them here."
        )
        return {}

    try:
        search_space = {}
        for name, param in config.items():
            if isinstance(param, (Categorical, Float, Integer, Function)):
                # The param is already a tune object and does not need to be parsed
                search_space[name] = param
                continue
            if isinstance(param, dict) and "grid_search" in param:
                # The param is already a tune grid search object and does not need to be parsed
                search_space[name] = param
                continue

            # From now on this function tries to parse the params from a dictionary
            if not isinstance(param, dict):
                raise ValueError(
                    f"Unable to parse '{param}' in the config as a tunable param."
                )

            # Convert specific keys to float if necessary
            for key in ["lower", "upper", "mean", "std"]:
                if key in param:
                    param[key] = float(param[key])

            param_type = param.pop("type")
            param = getattr(ray.tune, param_type)(**param)
            search_space[name] = param
        return search_space
    except Exception as exc:
        if hasattr(exc, "add_note"):
            # This was introduced from Python 3.11
            exc.add_note(
                f"{param} could not be set. Check that this parameter type is "
                "supported by Ray Tune at "
                "https://docs.ray.io/en/latest/tune/api/search_space.html"
            )
        raise exc
