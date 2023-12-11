"""
Provide functionalities to manage configuration files, including parsing,
execution, and dynamic override of fields.
"""

import sys
from typing import Dict, Any, Union, Optional
from jsonargparse import ArgumentParser, ActionConfigFile
import json
from omegaconf import OmegaConf
from pathlib import Path

from .components import BaseComponent
from .pipeline import Pipeline
from .utils import load_yaml, dynamically_import_class


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
    >>>       - class_path: itwinai.torch.trainer.TorchTrainerMG
    >>>         init_args:
    >>>           model:
    >>>             class_path: model.Net
    >>>           loss:
    >>>             class_path: torch.nn.NLLLoss
    >>>             init_args:
    >>>               reduction: mean

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

    config: Dict
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
        pipe_parser = ArgumentParser()
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
        step_name: str,
        pipeline_nested_key: str = "pipeline",
        verbose: bool = False
    ) -> BaseComponent:
        pipeline_dict = self.config
        for key in pipeline_nested_key.split('.'):
            pipeline_dict = pipeline_dict[key]

        step_dict_config = pipeline_dict['init_args']['steps'][step_name]

        if verbose:
            print(f"STEP '{step_name}' CONFIG:")
            print(json.dumps(step_dict_config, indent=4))

        # Wrap config under "step" field and parse it
        step_dict_config = {'step': step_dict_config}
        step_parser = ArgumentParser()
        step_parser.add_subclass_arguments(BaseComponent, "step")
        parsed_namespace = step_parser.parse_object(step_dict_config)
        return step_parser.instantiate_classes(parsed_namespace)["step"]


class ConfigParser2:
    """
    Deprecated: this pipeline structure does not allow for
    nested pipelines. However, it is more readable and the linking
    from name to step data could be achieved with OmegaConf. This
    could be reused in the future: left as example.

    Parses a configuration file, merging the steps into
    the pipeline and returning a pipeline object.
    It also provides functionalities for dynamic override
    of fields by means of nested key notation.

    Example:

    >>> # pipeline.yaml
    >>> pipeline:
    >>>   class_path: itwinai.pipeline.Pipeline
    >>>   steps: [server, client]
    >>>
    >>> server:
    >>>   class_path: mycode.ServerOptions
    >>>   init_args:
    >>>     host: localhost
    >>>     port: 80
    >>>
    >>> client:
    >>>   class_path: mycode.ClientOptions
    >>>   init_args:
    >>>     url: http://${server.init_args.host}:${server.init_args.port}/

    >>> from itwinai.parser import ConfigParser2
    >>>
    >>> parser = ConfigParser2(
    >>>     config='pipeline.yaml',
    >>>     override_keys={
    >>>         'server.init_args.port': 777
    >>>     }
    >>> )
    >>> pipeline = parser.parse_pipeline()
    >>> print(pipeline)
    >>> print(pipeline.steps)
    >>> print(pipeline.steps['server'].port)
    >>>
    >>> server = parser.parse_step('server')
    >>> print(server)
    >>> print(server.port)
    """

    config: Dict
    pipeline: Pipeline

    def __init__(
        self,
        config: Union[str, Dict],
        override_keys: Optional[Dict[str, Any]] = None
    ) -> None:
        self.config = config
        self.override_keys = override_keys
        if isinstance(self.config, str):
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
        pipe_parser = ArgumentParser()
        pipe_parser.add_subclass_arguments(Pipeline, pipeline_nested_key)
        pipe_dict = self.config[pipeline_nested_key]

        # Pop steps list from pipeline dictionary
        steps_list = pipe_dict['steps']
        del pipe_dict['steps']

        # Link steps with respective dictionaries
        if not pipe_dict.get('init_args'):
            pipe_dict['init_args'] = {}
        steps_dict = pipe_dict['init_args']['steps'] = {}
        for step_name in steps_list:
            steps_dict[step_name] = self.config[step_name]
        pipe_dict = {pipeline_nested_key: pipe_dict}

        if verbose:
            print("Assembled pipeline:")
            print(json.dumps(pipe_dict, indent=4))

        # Parse pipeline dict once merged with steps
        conf = pipe_parser.parse_object(pipe_dict)
        pipe = pipe_parser.instantiate_classes(conf)
        self.pipeline = pipe[pipeline_nested_key]
        return self.pipeline

    def parse_step(
        self,
        step_name: str,
        verbose: bool = False
    ) -> BaseComponent:
        step_dict_config = self.config[step_name]

        if verbose:
            print(f"STEP '{step_name}' CONFIG:")
            print(json.dumps(step_dict_config, indent=4))

        # Wrap config under "step" field and parse it
        step_dict_config = {'step': step_dict_config}
        step_parser = ArgumentParser()
        step_parser.add_subclass_arguments(BaseComponent, "step")
        parsed_namespace = step_parser.parse_object(step_dict_config)
        return step_parser.instantiate_classes(parsed_namespace)["step"]


class ItwinaiCLI:
    """
    Deprecated: the dynamic override does not work with nested parameters
    and may be confusing.

    CLI tool for executing a configuration file, with dynamic
    override of fields and variable interpolation with Omegaconf.

    Example:

    >>> # train.py
    >>> from itwinai.parser import ItwinaiCLI
    >>> cli = ItwinaiCLI()
    >>> cli.pipeline.execute()

    >>> # pipeline.yaml
    >>> pipeline:
    >>>   class_path: itwinai.pipeline.Pipeline
    >>>   steps: [server, client]
    >>>
    >>> server:
    >>>   class_path: mycode.ServerOptions
    >>>   init_args:
    >>>     host: localhost
    >>>     port: 80
    >>>
    >>> client:
    >>>   class_path: mycode.ClientOptions
    >>>   init_args:
    >>>     url: http://${server.init_args.host}:${server.init_args.port}/

    From command line:

    >>> python train.py --config itwinai-conf.yaml --help
    >>> python train.py --config itwinai-conf.yaml
    >>> python train.py --config itwinai-conf.yaml --server.port 8080
    """
    _parser: ArgumentParser
    _config: Dict
    pipeline: Pipeline

    def __init__(
        self,
        pipeline_nested_key: str = "pipeline",
        parser_mode: str = "omegaconf"
    ) -> None:
        self.pipeline_nested_key = pipeline_nested_key
        self.parser_mode = parser_mode
        self._init_parser()
        self._parser.add_argument(f"--{self.pipeline_nested_key}", type=dict)
        self._add_steps_arguments()
        self._config = self._parser.parse_args()

        # Merge steps into pipeline and parse it
        del self._config['config']
        pipe_parser = ConfigParser2(config=self._config.as_dict())
        self.pipeline = pipe_parser.parse_pipeline(
            pipeline_nested_key=self.pipeline_nested_key
        )

    def _init_parser(self):
        self._parser = ArgumentParser(parser_mode=self.parser_mode)
        self._parser.add_argument(
            "-c", "--config", action=ActionConfigFile,
            required=True,
            help="Path to a configuration file in json or yaml format."
        )

    def _add_steps_arguments(self):
        """Pre-parses the configuration file, dynamically adding all the
        component classes under 'steps' as arguments of the parser.
        """
        if "--config" not in sys.argv:
            raise ValueError(
                "--config parameter has to be specified with a "
                "valid path to a configuration file."
            )
        config_path = sys.argv.index("--config") + 1
        config_path = sys.argv[config_path]
        config = load_yaml(config_path)

        # Add steps to parser
        steps = filter(
            lambda itm: itm[0] != self.pipeline_nested_key,
            config.items()
        )
        steps = {
            step_name: step_data['class_path']
            for step_name, step_data in steps
        }

        for st_nested_key, step_class_str in steps.items():
            step_class = dynamically_import_class(step_class_str)
            self._add_step_arguments(
                step_class=step_class, nested_key=st_nested_key)

    def _add_step_arguments(self, step_class, nested_key):
        self._parser.add_subclass_arguments(
            baseclass=step_class, nested_key=nested_key)
