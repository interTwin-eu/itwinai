# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Provide functionalities to manage configuration files, including parsing,
execution, and dynamic override of fields.
"""

import ast
from pathlib import Path
from typing import Dict, Union

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .components import BaseComponent
from .pipeline import Pipeline


class ConfigParser:
    """Parses a pipeline from a configuration file.
    It also provides functionalities for dynamic override
    of fields by means of nested key notation.

    Args:
        config (Union[str, Dict]): path to YAML configuration file
            or dict storing a configuration.
        override_keys (Optional[Dict[str, Any]], optional): dict mapping
            nested keys to the value to override. Defaults to None.

    Example:


    >>> # pipeline.yaml file
    >>> pipeline:
    >>>   class_path: itwinai.pipeline.Pipeline
    >>>   init_args:
    >>>     steps:
    >>>       - class_path: dataloader.MNISTDataModuleTorch
    >>>         init_args:
    >>>           save_path: .tmp/
    >>>
    >>>       - class_path: itwinai.torch.trainer.TorchTrainer
    >>>         init_args:
    >>>           model:
    >>>             class_path: model.Net
    >>>
    >>> from itwinai.parser import ConfigParser
    >>>
    >>> parser = ConfigParser(
    >>>     config='pipeline.yaml',
    >>>     override_keys={
    >>>         'pipeline.init_args.steps.0.init_args.save_path': /save/path
    >>>     }
    >>> )
    >>> pipeline = parser.parse_pipeline()
    >>> print(pipeline)
    >>> print(pipeline.steps)
    >>>
    >>> dataloader = parser.parse_step(0)
    >>> print(dataloader)
    >>> print(dataloader.save_path)

    """

    #: Path to the yaml file containing the configuration to parse.
    config: str | Path
    #: A dictionary containing which keys to override in the configuration.
    pipeline: Pipeline

    def __init__(
        self,
        config: str | Path,
    ) -> None:
        self.config = config

    def parse_pipeline(
        self,
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict | None = None,
    ) -> Pipeline:
        import yaml  # for error handling

        # Load config from yaml
        try:
            raw_conf = OmegaConf.load(self.config)
        except yaml.scanner.ScannerError as e:
            e.add_note(
                f"Failed to load config from {self.config}! You might want to check "
                "the structure of your yaml file."
            )
            raise e

        if pipeline_nested_key not in raw_conf:
            raise ValueError(f"Pipeline key {pipeline_nested_key} not found.")

        # Override keys
        for override_key, override_value in override_keys.items():
            # if override_key not in raw_conf[pipeline_nested_key]:
            #     raise KeyError(
            #         f"Tried to override non-existing key {override_key}. Please make sure "
            #         "to only overwrite keys that are defined in your selected "
            #         f"pipeline: {pipeline_nested_key}."
            #     )
            inferred_type = ast.literal_eval(override_value)
            OmegaConf.update(raw_conf, override_key, inferred_type)

            print(
                f"Successfully overrode key {override_key}."
                f"It now has the value {inferred_type} of type {type(inferred_type)}."
            )

        conf = raw_conf[pipeline_nested_key]

        return conf

    def build_pipeline(
        self,
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict | None = None,
        steps: str | None = None,
        verbose: bool = False,
    ) -> Pipeline:
        """Parses the pipeline and instantiated all classes defined within it.

        Args:
            pipeline_nested_key (str, optional): nested key in the
                configuration file identifying the pipeline object.
                Defaults to "pipeline".
            verbose (bool): if True, prints the assembled pipeline
                to console formatted as JSON.

        Returns:
            Pipeline: instantiated pipeline.
        """
        conf = self.parse_pipeline(pipeline_nested_key, override_keys)

        if steps:
            conf.steps = self._get_selected_steps(steps, conf.steps)

        # Resolve interpolated parameters
        OmegaConf.resolve(conf)

        if verbose:
            print("Assembled pipeline:")
            print(OmegaConf.to_yaml(conf))

        pipeline = instantiate(conf)

        return pipeline

    def build_step(
        self,
        step_idx: Union[str, int],
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict | None = None,
        verbose: bool = False,
    ) -> BaseComponent:
        conf = self.parse_pipeline(pipeline_nested_key, override_keys)

        conf = conf.steps[step_idx]

        # Resolve interpolated parameters
        OmegaConf.resolve(conf)

        if verbose:
            print(f"Assembled step {step_idx} from {pipeline_nested_key}:")
            print(OmegaConf.to_yaml(conf))

        step = instantiate(conf)

        return step

    def _get_selected_steps(self, steps: str, conf_steps: list):
        # If steps is given as a slice
        if ":" in steps:
            try:
                slice_obj = slice(*[int(x) if x else None for x in steps.split(":")])
                return conf_steps[slice_obj]
            except ValueError:
                raise ValueError(f"Invalid slice notation: {steps}")

        # If steps is given as a single index
        elif steps.isdigit():
            index = int(steps)
            if 0 <= index < len(conf_steps):
                return [conf_steps[index]]
            else:
                raise IndexError(f"Step index out of range: {index}")

        # If steps is given as a name
        else:
            selected_steps = [step for step in conf_steps if step.get("_target_") == steps]
            if not selected_steps:
                raise ValueError(f"No steps found with name: {steps}")
            return selected_steps
