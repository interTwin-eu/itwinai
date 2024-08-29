"""
Provide functionalities to manage configuration files, including parsing,
execution, and dynamic override of fields.
"""

import logging
import os
from typing import Dict, Any, List, Type, Union, Optional
from jsonargparse import ArgumentParser as JAPArgumentParser
from jsonargparse import ActionConfigFile
from jsonargparse._formatters import DefaultHelpFormatter

import json
from omegaconf import OmegaConf
from pathlib import Path

from .components import BaseComponent
from .pipeline import Pipeline
from .utils import load_yaml


def add_replace_field(
    config: Dict,
    key_chain: str,
    value: Any
) -> None:
    """Replace or add (if not present) a field in a dictionary, following a
    path of dot-separated keys. Adding is not supported for list items.
    Inplace operation.

    Args:
        config (Dict): dictionary to be modified.
        key_chain (str): path of nested (dot-separated) keys to specify the
            location
            of the new value (e.g., 'foo.bar.line' adds/overwrites the value
            located at config['foo']['bar']['line']).
        value (Any): the value to insert.
    """
    sub_config = config
    for idx, k in enumerate(key_chain.split('.')):
        if idx >= len(key_chain.split('.')) - 1:
            # Last key reached
            break

        if isinstance(sub_config, (list, tuple)):
            k = int(k)
            next_elem = sub_config[k]
        else:
            next_elem = sub_config.get(k)

        if not isinstance(next_elem, (dict, list, tuple)):
            sub_config[k] = dict()

        sub_config = sub_config[k]
    if isinstance(sub_config, (list, tuple)):
        k = int(k)
    sub_config[k] = value


class ConfigParser:
    """
    Parses a pipeline from a configuration file.
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

    #: Configuration to parse.
    config: Dict
    #: Pipeline object instantiated from the configuration file.
    pipeline: Pipeline

    def __init__(
        self,
        config: Union[str, Dict],
        override_keys: Optional[Dict[str, Any]] = None
    ) -> None:
        self.config = config
        self.override_keys = override_keys
        if isinstance(self.config, (str, Path)):
            self.config = load_yaml(self.config)
        self._dynamic_override_keys()
        self._omegaconf_interpolate()

    def _dynamic_override_keys(self):
        if self.override_keys is not None:
            for key_chain, value in self.override_keys.items():
                add_replace_field(self.config, key_chain, value)

    def _omegaconf_interpolate(self) -> None:
        """Performs variable interpolation with OmegaConf on internal
        configuration file.
        """
        conf = OmegaConf.create(self.config)
        self.config = OmegaConf.to_container(conf, resolve=True)

    def parse_pipeline(
        self,
        pipeline_nested_key: str = "pipeline",
        verbose: bool = False
    ) -> Pipeline:
        """Merges steps into pipeline and parses it.

        Args:
            pipeline_nested_key (str, optional): nested key in the
                configuration file identifying the pipeline object.
                Defaults to "pipeline".
            verbose (bool): if True, prints the assembled pipeline
                to console formatted as JSON.

        Returns:
            Pipeline: instantiated pipeline.
        """
        pipe_parser = JAPArgumentParser()
        pipe_parser.add_subclass_arguments(Pipeline, "pipeline")

        pipe_dict = self.config
        for key in pipeline_nested_key.split('.'):
            pipe_dict = pipe_dict[key]
        # pipe_dict = self.config[pipeline_nested_key]
        pipe_dict = {"pipeline": pipe_dict}

        if verbose:
            print("Assembled pipeline:")
            print(json.dumps(pipe_dict, indent=4))

        # Parse pipeline dict once merged with steps
        conf = pipe_parser.parse_object(pipe_dict)
        pipe = pipe_parser.instantiate_classes(conf)
        self.pipeline = pipe["pipeline"]
        return self.pipeline

    def parse_step(
        self,
        step_idx: Union[str, int],
        pipeline_nested_key: str = "pipeline",
        verbose: bool = False
    ) -> BaseComponent:
        pipeline_dict = self.config
        for key in pipeline_nested_key.split('.'):
            pipeline_dict = pipeline_dict[key]

        step_dict_config = pipeline_dict['init_args']['steps'][step_idx]

        if verbose:
            print(f"STEP '{step_idx}' CONFIG:")
            print(json.dumps(step_dict_config, indent=4))

        # Wrap config under "step" field and parse it
        step_dict_config = {'step': step_dict_config}
        step_parser = JAPArgumentParser()
        step_parser.add_subclass_arguments(BaseComponent, "step")
        parsed_namespace = step_parser.parse_object(step_dict_config)
        return step_parser.instantiate_classes(parsed_namespace)["step"]


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
            *args, env_prefix=env_prefix, formatter_class=formatter_class,
            exit_on_error=exit_on_error, logger=logger, version=version,
            print_config=print_config, parser_mode=parser_mode,
            dump_header=dump_header, default_config_files=default_config_files,
            default_env=default_env,
            default_meta=default_meta, **kwargs)
        self.add_argument(
            "-c", "--config", action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format."
        )

# type: ignore
