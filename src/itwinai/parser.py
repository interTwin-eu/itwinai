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

import logging
import os
from typing import List, Optional, Type, Union

from jsonargparse import ActionConfigFile
from jsonargparse import ArgumentParser as JAPArgumentParser
from jsonargparse._formatters import DefaultHelpFormatter


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
