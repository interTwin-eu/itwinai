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
from omegaconf import OmegaConf, dictconfig, errors, listconfig

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
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict = {},
        steps: List[str] | List[int] | None = None,
        verbose: bool = False,
    ) -> Pipeline:
        """Parses the pipeline and instantiated all classes defined within it.

        Args:
            pipeline_nested_key (str, optional): Nested key in the configuration file
                identifying the pipeline object. Defaults to "pipeline".
            override_keys ([Dict[str, Any]]): A dict mapping
                nested keys to the value to override. Defaults to {}.
            steps ((list(str) | list(int)), optional): If building a pipeline, allows you to
                select which step(s) to include in the pipeline. Accepted values are lists
                of indices for config files where the steps are defined as lists, or lists of
                names of steps, if they are defined in the configuration as a dict.
                Defaults to None.
            verbose (bool): if True, prints the assembled pipeline
                to console formatted as JSON.

        Returns:
            Pipeline | BaseComponent: instantiated pipeline or component.
        """
        conf = self.parse_pipeline(pipeline_nested_key, override_keys)

        # Select steps
        if steps:
            conf = self._select_steps(conf, steps)

        # Resolve interpolated parameters
        OmegaConf.resolve(conf)

        if verbose:
            print(f"Assembled {type} from {pipeline_nested_key}:")
            print(OmegaConf.to_yaml(conf))

        return instantiate(conf)

    def parse_pipeline(
        self,
        pipeline_nested_key: str = "pipeline",
        override_keys: Dict = {},
    ) -> dictconfig.DictConfig:
        """Parses the pipeline from a yaml file into an OmegaConf DictConfig.

        Args:
            pipeline_nested_key (str, optional): Nested key in the configuration file
                identifying the pipeline object. Defaults to "pipeline".
            override_keys (Dict): A dict mapping
                nested keys to the value to override. Defaults to {}.

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

        # Override keys
        for override_key, override_value in override_keys.items():
            inferred_type = ast.literal_eval(override_value)
            OmegaConf.update(raw_conf, override_key, inferred_type)

            print(
                f"Successfully overrode key {override_key}."
                f"It now has the value {inferred_type} of type {type(inferred_type)}."
            )
        try:
            conf = OmegaConf.select(raw_conf, pipeline_nested_key, throw_on_missing=True)
        except Exception as e:
            e.add_note(f"Could not find pipeline key {pipeline_nested_key} in config.")
            raise e

        return conf

    def _select_steps(
        self, conf: listconfig.Listconfig | dictconfig.DictConfig, steps: List[int] | List[str]
    ):
        """Selects the steps given from the configuration object.
        If only one step is selected, returns a configuration with only that step. Otherwise
        returns a pipeline with all the selected steps as a list.

        Args:
            conf (listconfig.Listconfig | dictconfig.DictConfig):
                The configuration of the pipeline
            steps (List[int] | List[str]): The list of steps

        Returns:
            listconfig.Listconfig | dictconfig.DictConfig: The updated configuration
        """
        if len(steps) == 1:
            return OmegaConf.create(conf.steps[steps[0]])

        selected_steps = [conf.steps[step] for step in steps]
        OmegaConf.update(conf, conf.steps, selected_steps)

        return conf


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
