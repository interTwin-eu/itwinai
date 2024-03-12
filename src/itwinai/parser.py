"""
Provide functionalities to manage configuration files, including parsing,
execution, and dynamic override of fields.
"""

import logging
import os
from typing import List, Type, Union, Optional
from jsonargparse import ArgumentParser as JAPArgumentParser
from jsonargparse import ActionConfigFile
from jsonargparse._formatters import DefaultHelpFormatter


class ArgumentParser(JAPArgumentParser):
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
        """Initializer for ArgumentParser instance.

        All the arguments from the initializer of `argparse.ArgumentParser
        <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>`_
        are supported. Additionally it accepts:

        Args:
            env_prefix: Prefix for environment variables. ``True`` to derive
            from ``prog``.
            formatter_class: Class for printing help messages.
            logger: Configures the logger, see :class:`.LoggerProperty`.
            version: Program version which will be printed by the --version
            argument.
            print_config: Add this as argument to print config, set None to
            disable.
            parser_mode: Mode for parsing config files: ``'yaml'``,
            ``'jsonnet'`` or ones added via :func:`.set_loader`.
            dump_header: Header to include as comment when dumping a config
            object.
            default_config_files: Default config file locations, e.g.
            :code:`['~/.config/myapp/*.yaml']`.
            default_env: Set the default value on whether to parse environment
            variables.
            default_meta: Set the default value on whether to include metadata
            in config objects.
        """
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
