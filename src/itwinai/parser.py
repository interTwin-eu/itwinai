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
        """Initializer for ArgumentParser instance. It can parse arguments from
        a series of configuration files. Example:

        >>> python main.py --config base-conf.yaml --config other-conf.yaml \\
        >>> --param OVERRIDE_VAL

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


# class ConfigParser2:
#     """
#     Deprecated: this pipeline structure does not allow for
#     nested pipelines. However, it is more readable and the linking
#     from name to step data could be achieved with OmegaConf. This
#     could be reused in the future: left as example.

#     Parses a configuration file, merging the steps into
#     the pipeline and returning a pipeline object.
#     It also provides functionalities for dynamic override
#     of fields by means of nested key notation.

#     Example:

#     >>> # pipeline.yaml
#     >>> pipeline:
#     >>>   class_path: itwinai.pipeline.Pipeline
#     >>>   steps: [server, client]
#     >>>
#     >>> server:
#     >>>   class_path: mycode.ServerOptions
#     >>>   init_args:
#     >>>     host: localhost
#     >>>     port: 80
#     >>>
#     >>> client:
#     >>>   class_path: mycode.ClientOptions
#     >>>   init_args:
#     >>>     url: http://${server.init_args.host}:${server.init_args.port}/

#     >>> from itwinai.parser import ConfigParser2
#     >>>
#     >>> parser = ConfigParser2(
#     >>>     config='pipeline.yaml',
#     >>>     override_keys={
#     >>>         'server.init_args.port': 777
#     >>>     }
#     >>> )
#     >>> pipeline = parser.parse_pipeline()
#     >>> print(pipeline)
#     >>> print(pipeline.steps)
#     >>> print(pipeline.steps['server'].port)
#     >>>
#     >>> server = parser.parse_step('server')
#     >>> print(server)
#     >>> print(server.port)
#     """

#     config: Dict
#     pipeline: Pipeline

#     def __init__(
#         self,
#         config: Union[str, Dict],
#         override_keys: Optional[Dict[str, Any]] = None
#     ) -> None:
#         self.config = config
#         self.override_keys = override_keys
#         if isinstance(self.config, str):
#             self.config = load_yaml(self.config)
#         self._dynamic_override_keys()
#         self._omegaconf_interpolate()

#     def _dynamic_override_keys(self):
#         if self.override_keys is not None:
#             for key_chain, value in self.override_keys.items():
#                 add_replace_field(self.config, key_chain, value)

#     def _omegaconf_interpolate(self) -> None:
#         """Performs variable interpolation with OmegaConf on internal
#         configuration file.
#         """
#         conf = OmegaConf.create(self.config)
#         self.config = OmegaConf.to_container(conf, resolve=True)

#     def parse_pipeline(
#         self,
#         pipeline_nested_key: str = "pipeline",
#         verbose: bool = False
#     ) -> Pipeline:
#         """Merges steps into pipeline and parses it.

#         Args:
#             pipeline_nested_key (str, optional): nested key in the
#             configuration file identifying the pipeline object.
#             Defaults to "pipeline".
#             verbose (bool): if True, prints the assembled pipeline
#             to console formatted as JSON.

#         Returns:
#             Pipeline: instantiated pipeline.
#         """
#         pipe_parser = JAPArgumentParser()
#         pipe_parser.add_subclass_arguments(Pipeline, pipeline_nested_key)
#         pipe_dict = self.config[pipeline_nested_key]

#         # Pop steps list from pipeline dictionary
#         steps_list = pipe_dict['steps']
#         del pipe_dict['steps']

#         # Link steps with respective dictionaries
#         if not pipe_dict.get('init_args'):
#             pipe_dict['init_args'] = {}
#         steps_dict = pipe_dict['init_args']['steps'] = {}
#         for step_name in steps_list:
#             steps_dict[step_name] = self.config[step_name]
#         pipe_dict = {pipeline_nested_key: pipe_dict}

#         if verbose:
#             print("Assembled pipeline:")
#             print(json.dumps(pipe_dict, indent=4))

#         # Parse pipeline dict once merged with steps
#         conf = pipe_parser.parse_object(pipe_dict)
#         pipe = pipe_parser.instantiate_classes(conf)
#         self.pipeline = pipe[pipeline_nested_key]
#         return self.pipeline

#     def parse_step(
#         self,
#         step_name: str,
#         verbose: bool = False
#     ) -> BaseComponent:
#         step_dict_config = self.config[step_name]

#         if verbose:
#             print(f"STEP '{step_name}' CONFIG:")
#             print(json.dumps(step_dict_config, indent=4))

#         # Wrap config under "step" field and parse it
#         step_dict_config = {'step': step_dict_config}
#         step_parser = JAPArgumentParser()
#         step_parser.add_subclass_arguments(BaseComponent, "step")
#         parsed_namespace = step_parser.parse_object(step_dict_config)
#         return step_parser.instantiate_classes(parsed_namespace)["step"]


# class ItwinaiCLI2:
#     """
#     Deprecated: the dynamic override does not work with nested parameters
#     and may be confusing.

#     CLI tool for executing a configuration file, with dynamic
#     override of fields and variable interpolation with Omegaconf.

#     Example:

#     >>> # train.py
#     >>> from itwinai.parser import ItwinaiCLI
#     >>> cli = ItwinaiCLI()
#     >>> cli.pipeline.execute()

#     >>> # pipeline.yaml
#     >>> pipeline:
#     >>>   class_path: itwinai.pipeline.Pipeline
#     >>>   steps: [server, client]
#     >>>
#     >>> server:
#     >>>   class_path: mycode.ServerOptions
#     >>>   init_args:
#     >>>     host: localhost
#     >>>     port: 80
#     >>>
#     >>> client:
#     >>>   class_path: mycode.ClientOptions
#     >>>   init_args:
#     >>>     url: http://${server.init_args.host}:${server.init_args.port}/

#     From command line:

#     >>> python train.py --config itwinai-conf.yaml --help
#     >>> python train.py --config itwinai-conf.yaml
#     >>> python train.py --config itwinai-conf.yaml --server.port 8080
#     """
#     _parser: JAPArgumentParser
#     _config: Dict
#     pipeline: Pipeline

#     def __init__(
#         self,
#         pipeline_nested_key: str = "pipeline",
#         parser_mode: str = "omegaconf"
#     ) -> None:
#         self.pipeline_nested_key = pipeline_nested_key
#         self.parser_mode = parser_mode
#         self._init_parser()
#         self._parser.add_argument(f"--{self.pipeline_nested_key}", type=dict)
#         self._add_steps_arguments()
#         self._config = self._parser.parse_args()

#         # Merge steps into pipeline and parse it
#         del self._config['config']
#         pipe_parser = ConfigParser2(config=self._config.as_dict())
#         self.pipeline = pipe_parser.parse_pipeline(
#             pipeline_nested_key=self.pipeline_nested_key
#         )

#     def _init_parser(self):
#         self._parser = JAPArgumentParser(parser_mode=self.parser_mode)
#         self._parser.add_argument(
#             "-c", "--config", action=ActionConfigFile,
#             required=True,
#             help="Path to a configuration file in json or yaml format."
#         )

#     def _add_steps_arguments(self):
#         """Pre-parses the configuration file, dynamically adding all the
#         component classes under 'steps' as arguments of the parser.
#         """
#         if "--config" not in sys.argv:
#             raise ValueError(
#                 "--config parameter has to be specified with a "
#                 "valid path to a configuration file."
#             )
#         config_path = sys.argv.index("--config") + 1
#         config_path = sys.argv[config_path]
#         config = load_yaml(config_path)

#         # Add steps to parser
#         steps = filter(
#             lambda itm: itm[0] != self.pipeline_nested_key,
#             config.items()
#         )
#         steps = {
#             step_name: step_data['class_path']
#             for step_name, step_data in steps
#         }

#         for st_nested_key, step_class_str in steps.items():
#             step_class = dynamically_import_class(step_class_str)
#             self._add_step_arguments(
#                 step_class=step_class, nested_key=st_nested_key)

#     def _add_step_arguments(self, step_class, nested_key):
#         self._parser.add_subclass_arguments(
#             baseclass=step_class, nested_key=nested_key)
