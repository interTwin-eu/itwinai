"""
Provide functionalities to manage configuration files, including parsing,
execution, and dynamic override of fields.
"""

from typing import Any
from jsonargparse import ArgumentParser, ActionConfigFile, Namespace

from .components import BaseComponent


class ItwinaiCLI:
    _parser: ArgumentParser
    pipeline: BaseComponent

    def __init__(
        self,
        pipeline_nested_key: str = "pipeline",
        args: Any = None,
        parser_mode: str = "omegaconf"
    ) -> None:
        self.pipeline_nested_key = pipeline_nested_key
        self.args = args
        self.parser_mode = parser_mode
        self._init_parser()
        self._parse_args()
        pipeline_inst = self._parser.instantiate_classes(self._config)
        self.pipeline = pipeline_inst[self.pipeline_nested_key]

    def _init_parser(self):
        self._parser = ArgumentParser(parser_mode=self.parser_mode)
        self._parser.add_argument(
            "-c", "--config", action=ActionConfigFile,
            required=True,
            help="Path to a configuration file in json or yaml format."
        )
        self._parser.add_subclass_arguments(
            baseclass=BaseComponent,
            nested_key=self.pipeline_nested_key
        )

    def _parse_args(self):
        if isinstance(self.args, (dict, Namespace)):
            self._config = self._parser.parse_object(self.args)
        else:
            self._config = self._parser.parse_args(self.args)
