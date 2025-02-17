"""Logic to parse configuration and transform it into Ray objects."""

from pathlib import Path
from typing import Dict

import ray.train
import ray.tune


def tune_config(tune_config: Dict | None) -> ray.tune.TuneConfig | None:
    from .tuning import get_raytune_scheduler, get_raytune_search_alg

    if not tune_config:
        print(
            "WARNING: Empty Tune Config configured. Using the default configuration with "
            "a single trial."
        )
        return

    search_alg = get_raytune_search_alg(tune_config)
    scheduler = get_raytune_scheduler(tune_config)

    metric = tune_config.get("metric", "loss")
    mode = tune_config.get("mode", "min")

    try:
        return ray.tune.TuneConfig(
            **tune_config,
            search_alg=search_alg,
            scheduler=scheduler,
            metric=metric,
            mode=mode,
        )
    except AttributeError as exc:
        exc.add_note(
            "Could not instantiate TuneConfig. Please ensure that you have passed the "
            "correct arguments for it. You can find more information for which "
            "arguments to set at "
            "https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html."
        )
        raise exc


def scaling_config(scaling_config: Dict | None) -> ray.train.ScalingConfig | None:
    if not scaling_config:
        print("WARNING: No Scaling Config configured. Running trials non-distributed.")
        return

    try:
        scaling_config = ray.train.ScalingConfig(**scaling_config)
    except AttributeError as exc:
        exc.add_note(
            "Could not instantiate ScalingConfig. Please ensure that you have passed the "
            "correct arguments for it. You can find more information for which "
            "arguments to set at "
            "https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html"
        )
        raise exc


def run_config(run_config: Dict | None) -> ray.train.RunConfig | None:
    if not run_config:
        print("WARNING: No RunConfig provided. Assuming local or single-node execution.")
        return

    try:
        storage_path = Path(run_config.pop("storage_path")).resolve()

        if not storage_path:
            print("INFO: Empty storage path provided. Using default path 'ray_checkpoints'")
            storage_path = Path("ray_checkpoints").resolve()

        run_config = ray.train.RunConfig(**run_config, storage_path=storage_path)
    except AttributeError as exc:
        exc.add_note(
            "Could not instantiate RunConfig. Please ensure that you have passed the "
            "correct arguments for it. You can find more information for which "
            "arguments to set at "
            "https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html"
        )
        raise exc


def search_space(config: Dict | None) -> Dict:
    if not config:
        print(
            "WARNING: No training_loop_config detected. "
            "If you want to tune any hyperparameters, make sure to define them here."
        )
        return {}

    try:
        search_space = {}
        for name, param in config.items():
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
    except AttributeError as exc:
        exc.add_note(
            f"{param} could not be set. Check that this parameter type is "
            "supported by Ray Tune at "
            "https://docs.ray.io/en/latest/tune/api/search_space.html"
        )
        raise exc
