# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch
# --------------------------------------------------------------------------------------

"""Provide functionalities to manage configuration files, including parsing,
execution, and dynamic override of fields.
"""

import ast
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, Union

from hydra.utils import instantiate
from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser as JAPArgumentParser
from jsonargparse._formatters import DefaultHelpFormatter
from omegaconf import OmegaConf, dictconfig

from .pipeline import Pipeline


class ConfigParser:
    """Parses a pipeline from a configuration file and creates an instantiated pipeline or
    BaseComponent object. It also provides functionalities for dynamic override of fields.

    Args:
        config (Union[str, Path]): path to YAML configuration file.

    Example:
    >>> # pipeline.yaml file
    >>>
    >>> save_path=.tmp/
    >>> pipeline:
    >>>   _target_: itwinai.pipeline.Pipeline
    >>>   steps:
    >>>     - _target_: dataloader.MNISTDataModuleTorch
    >>>       save_path: {$save_path}
    >>>
    >>>     - _target_: itwinai.torch.trainer.TorchTrainer
    >>>       model:
    >>>         _target_: model.Net
    >>>
    >>> from itwinai.parser import ConfigParser
    >>>
    >>> parser = ConfigParser(config='pipeline.yaml')
    >>> pipeline = parser.build_from_config(
    >>>     override_keys={
    >>>         'save_path': /save/path
    >>>     }
    >>> )
    >>> print(pipeline)
    >>> print(pipeline.steps)
    >>>
    >>> dataloader = parser.build_from_config(
    >>>     type="step",
            override_keys={
    >>>         'save_path': /save/path
    >>>     },
    >>>     step_idx=0
    >>> )
    >>> print(dataloader)
    >>> print(dataloader.save_path)
    """

    #: Path to the yaml file containing the configuration to parse.
    config: str | Path

    def __init__(
        self,
        config: str | Path,
    ) -> None:
        self.config = config

    def build_from_config(
        self,
        type: Literal["pipeline", "step"] = "pipeline",
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict | None = None,
        steps: str | None = None,
        step_idx: Union[str, int] | None = None,
        verbose: bool = False,
    ) -> Pipeline:
        """Parses the pipeline and instantiated all classes defined within it.

        Args:
            type (Literal["pipeline", "step"]): The type of object to build. If set to "step",
                only the object(s) defined by the step given by 'step_idx' is built.
                Defaults to "pipeline".
            pipeline_nested_key (str, optional): Nested key in the configuration file
                identifying the pipeline object. Defaults to "pipeline".
            override_keys ([Dict[str, Any]], optional): A dict mapping
                nested keys to the value to override. Defaults to None.
            steps (str, optional): If building a pipeline, allows you to select which step(s)
                to include in the pipeline. Accepted values are indices,
                python slices (e.g., 0:3 or 2:10:100), and string names of steps.
                Defaults to None.
            step_idx (Union[str, int], optional): If building only a step, used to identify
                which one. Must be set if 'type' is set to 'step'. Defaults to None.
            verbose (bool): if True, prints the assembled pipeline
                to console formatted as JSON.

        Returns:
            Pipeline | BaseComponent: instantiated pipeline or component.
        """
        conf = self.parse_pipeline(pipeline_nested_key, override_keys)

        # Select steps
        if type == "pipeline":
            if steps:
                conf.steps = self._get_selected_steps(steps, conf.steps)
        else:
            conf = conf.steps[step_idx]

        # Resolve interpolated parameters
        OmegaConf.resolve(conf)

        if verbose:
            print(f"Assembled {type} from {pipeline_nested_key}:")
            print(OmegaConf.to_yaml(conf))

        return instantiate(conf)

    def parse_pipeline(
        self,
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict | None = None,
    ) -> dictconfig.DictConfig:
        """Parses the pipeline from a yaml file into an OmegaConf DictConfig.

        Args:
            pipeline_nested_key (str, optional): Nested key in the configuration file
                identifying the pipeline object. Defaults to "pipeline".
            override_keys (Dict | None, optional): A dict mapping
                nested keys to the value to override. Defaults to None.

        Raises:
            e: Failed to load config from yaml. Most likely due to a badly structured
                configuration.
            ValueError: pipeline_nested_key not present in configuration file.

        Returns:
            dictconfig.DictConfig: The parsed configuration.
        """
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
            inferred_type = ast.literal_eval(override_value)
            OmegaConf.update(raw_conf, override_key, inferred_type)

            print(
                f"Successfully overrode key {override_key}."
                f"It now has the value {inferred_type} of type {type(inferred_type)}."
            )

        conf = raw_conf[pipeline_nested_key]

        return conf

    def _get_selected_steps(self, steps: str, conf_steps: list):
        """Selects the steps of the pipeline to be executed.

        Args:
            steps (str): Selects the steps of the pipeline. Accepted values are indices,
                python slices (e.g., 0:3 or 2:10:100), and string names of steps.
            conf_steps (list): A list of all the steps in the pipeline configuration.

        Raises:
            ValueError: Invalid slice notation
            IndexError: Index out of range
            ValueError: Invalid step name given

        Returns:
            list: The steps selected from the pipeline.
        """
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


class ArgumentParser(JAPArgumentParser):
    """Wrapper of ``jsonargparse.ArgumentParser``.
    Initializer for ArgumentParser instance. It can parse arguments from
    a series of configuration files. Example:

    >>> python main.py --config base-conf.yaml --config other-conf.yaml \\
    >>> --param OVERRIDE_VAL

    All the arguments from the initializer of `argparse.ArgumentParser
    <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
    are supported. Additionally it accepts:

    Args:
        env_prefix (Union[bool, str], optional): Prefix for environment
            variables. ``True`` to derive from ``prog``.. Defaults to True.
        formatter_class (Type[DefaultHelpFormatter], optional): Class for
            printing help messages. Defaults to DefaultHelpFormatter.
        exit_on_error (bool, optional): Defaults to True.
        logger (Union[bool, str, dict, logging.Logger], optional):
            Configures the logger, see :class:`.LoggerProperty`.
            Defaults to False.
        version (Optional[str], optional): Program version which will be
            printed by the --version argument. Defaults to None.
        print_config (Optional[str], optional): Add this as argument to
            print config, set None to disable.
            Defaults to "--print_config".
        parser_mode (str, optional): Mode for parsing config files:
            ``'yaml'``, ``'jsonnet'`` or ones added via
            :func:`.set_loader`.. Defaults to "yaml".
        dump_header (Optional[List[str]], optional): Header to include
            as comment when dumping a config object. Defaults to None.
        default_config_files
            (Optional[List[Union[str, os.PathLike]]], optional):
            Default config file locations, e.g.
            :code:`['~/.config/myapp/*.yaml']`. Defaults to None.
        default_env (bool, optional): Set the default value on whether
            to parse environment variables. Defaults to False.
        default_meta (bool, optional): Set the default value on whether
            to include metadata in config objects. Defaults to True.
    """

    def __init__(
        self,
        *args,
        env_prefix: Union[bool, str] = True,
        formatter_class: Type[DefaultHelpFormatter] = DefaultHelpFormatter,
        exit_on_error: bool = True,
        logger: Union[bool, str, dict, logging.Logger] = False,
        version: Optional[str] = None,
        print_config: Optional[str] = "--print_config",
        parser_mode: str = "yaml",
        dump_header: Optional[List[str]] = None,
        default_config_files: Optional[List[Union[str, os.PathLike]]] = None,
        default_env: bool = False,
        default_meta: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            env_prefix=env_prefix,
            formatter_class=formatter_class,
            exit_on_error=exit_on_error,
            logger=logger,
            version=version,
            print_config=print_config,
            parser_mode=parser_mode,
            dump_header=dump_header,
            default_config_files=default_config_files,
            default_env=default_env,
            default_meta=default_meta,
            **kwargs,
        )
        self.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
