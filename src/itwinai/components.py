from __future__ import annotations
from typing import Iterable, Dict, Any, Optional, Tuple, Union
from abc import ABCMeta, abstractmethod
import time
from jsonargparse import ArgumentParser

# import logging
# from logging import Logger as PythonLogger

from .cluster import ClusterEnvironment
from .types import ModelML, DatasetML
from .serialization import ModelLoader
from .utils import load_yaml


class Executable(metaclass=ABCMeta):
    """Base Executable class.

        Args:
            name (Optional[str], optional): unique identifier for a step.
                Defaults to None.
            logs_path (Optional[str], optional): where to store the logs
                produced by Python logging. Defaults to None.
        """
    name: str = 'unnamed'
    is_setup: bool = False
    cluster: ClusterEnvironment = None
    parent: Executor = None
    # logs_dir: str = None
    # log_file: str = None
    # console: PythonLogger = None

    def __init__(
        self,
        name: Optional[str] = None,
        # logs_dir: Optional[str] = None,
        # debug: bool = False,
        **kwargs
    ) -> None:
        self.name = name if name is not None else self.__class__.__name__
        # self.logs_dir = logs_dir
        # self.debug = debug

    def __call__(
        self,
        *args: Any,
        config: Optional[Dict] = None,
        **kwargs: Any
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        # WAIT! This method SHOULD NOT be overridden. This is just a wrapper.
        # Override execute() instead!
        msg = f"Starting execution of '{self.name}'..."
        self._printout(msg)
        start_t = time.time()
        try:
            # print(f'ARGS: {args}')
            # print(f'KWARGS: {kwargs}')
            result = self.execute(*args, **kwargs, config=config)
        finally:
            self.cleanup()
        self.exec_t = time.time() - start_t
        msg = f"'{self.name}' executed in {self.exec_t:.3f}s"
        self._printout(msg)
        return result

    @abstractmethod
    def execute(
        self,
        *args,
        config: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        """"Execute some operations.

        Args:
            args (Any, optional): generic input of the executable step.
            config (Dict, optional): key-value configuration.
                Defaults to None.

        Returns:
            Tuple[Optional[Tuple], Optional[Dict]]: tuple structured as
                (results, config).
        """
        return args, config

    # def setup_console(self):
    #     """Setup Python logging"""
    #     self.log_file = os.path.join(self.logs_dir, self.name + ".log")
    #     f_handler = logging.FileHandler(self.log_file, mode='w')
    #     stdout_h = logging.StreamHandler(sys.stdout)

    #     if self.debug:
    #         log_format = ("%(asctime)s %(levelname)s "
    #                       "[%(filename)s:%(lineno)s - %(funcName)s()]: "
    #                       "%(message)s")
    #     else:
    #         log_format = ("%(levelname)s : %(message)s")

    #     logging.basicConfig(
    #         level=logging.DEBUG if self.debug else logging.INFO,
    #         handlers=[f_handler, stdout_h],
    #         format=log_format,
    #         datefmt="%Y-%m-%d %H:%M:%S"
    #     )
    #     self.console = logging.getLogger(self.name)

    def setup(self, parent: Optional[Executor] = None) -> None:
        """Inherit properties from parent Executor instance.

        Args:
            parent (Optional[Executor], optional): parent executor.
                Defaults to None.
        """
        if parent is None:
            # # Setup Python logging ("console")
            # self.logs_dir = '.logs'
            # os.makedirs(self.logs_dir, exist_ok=True)
            # self.setup_console()
            self.is_setup = True
            return
        if self.cluster is None:
            self.cluster = parent.cluster

        # # Python logging ("console")
        # if self.logs_dir is None:
        #     self.logs_dir = parent.logs_dir
        # if self.log_file is None:
        #     self.log_file = parent.log_file
        # if self.console is None:
        #     self.console = logging.getLogger(self.name)

        self.is_setup = True

    def cleanup(self):
        pass

    def _printout(self, msg: str):
        msg = f"# {msg} #"
        print("#"*len(msg))
        print(msg)
        print("#"*len(msg))


class Trainer(Executable):
    """Trains a machine learning model."""
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass


class Predictor(Executable):
    """Applies a pre-trained machine learning model to unseen data."""

    model: ModelML

    def __init__(
        self,
        model: Union[ModelML, ModelLoader],
        name: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self.model = model() if isinstance(model, ModelLoader) else model

    def execute(
        self,
        predict_dataset: DatasetML,
        config: Optional[Dict] = None,
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        """"Execute some operations.

        Args:
            predict_dataset (DatasetML): dataset object for inference.
            config (Dict, optional): key-value configuration.
                Defaults to None.

        Returns:
            Tuple[Optional[Tuple], Optional[Dict]]: tuple structured as
                (results, config).
        """
        return self.predict(predict_dataset), config

    @abstractmethod
    def predict(
        self,
        predict_dataset: DatasetML,
        model: Optional[ModelML] = None
    ) -> Iterable[Any]:
        """Applies a machine learning model on a dataset of samples.

        Args:
            predict_dataset (DatasetML): dataset for inference.
            model (Optional[ModelML], optional): overrides the internal model,
                if given. Defaults to None.

        Returns:
            Iterable[Any]: predictions with the same cardinality of the
                input dataset.
        """


class DataGetter(Executable):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass


class DataPreproc(Executable):
    @abstractmethod
    def preproc(self, *args, **kwargs):
        pass


# class StatGetter(Executable):
#     @abstractmethod
#     def stats(self, *args, **kwargs):
#         pass


class Saver(Executable):
    @abstractmethod
    def save(self, *args, **kwargs):
        pass


class Executor(Executable):
    """Sets-up and executes a sequence of Executable steps."""

    steps: Union[Dict[str, Executable], Iterable[Executable]]
    constructor_args: Dict

    def __init__(
        self,
        steps: Union[Dict[str, Executable], Iterable[Executable]],
        name: Optional[str] = None,
        # logs_dir: Optional[str] = None,
        # debug: bool = False,
        **kwargs
    ):
        # super().__init__(name=name, logs_dir=logs_dir, debug=debug, **kwargs)
        super().__init__(name=name, **kwargs)
        self.steps = steps
        self.constructor_args = kwargs

    def __getitem__(self, subscript: Union[str, int, slice]) -> Executor:
        if isinstance(subscript, slice):
            # First, convert to list if is a dict
            if isinstance(self.steps, dict):
                steps = list(self.steps.items())
            else:
                steps = self.steps
            # Second, perform slicing
            s = steps[subscript.start:subscript.stop: subscript.step]
            # Third, reconstruct dict, if it is a dict
            if isinstance(self.steps, dict):
                s = dict(s)
            # Fourth, return sliced sub-pipeline, preserving its
            # initial structure
            sliced = self.__class__(
                steps=s,
                **self.constructor_args
            )
            return sliced
        else:
            return self.steps[subscript]

    def __len__(self) -> int:
        return len(self.steps)

    def setup(self, parent: Optional[Executor] = None) -> None:
        """Inherit properties from parent Executor instance, then
        propagates its properties to its own child steps.

        Args:
            parent (Optional[Executor], optional): parent executor.
                Defaults to None.
        """
        super().setup(parent)
        if isinstance(self.steps, dict):
            steps = list(self.steps.values())
        else:
            steps = self.steps

        for step in steps:
            step.setup(self)
            step.is_setup = True

    # def setup(self, config: Dict = None):
    #     """Pass a key-value based configuration down the pipeline,
    #     to propagate information computed at real-time.

    #     Args:
    #         config (Dict, optional): key-value configuration.
    #           Defaults to None.
    #     """
    #     for step in self.steps:
    #         config = step.setup(config)

    def execute(
        self,
        *args,
        config: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        """"Execute some operations.

        Args:
            args (Tuple, optional): generic input of the first executable step
                in the pipeline.
            config (Dict, optional): key-value configuration.
                Defaults to None.

        Returns:
            Tuple[Optional[Tuple], Optional[Dict]]: tuple structured as
                (results, config).
        """
        if isinstance(self.steps, dict):
            steps = list(self.steps.values())
        else:
            steps = self.steps

        for step in steps:
            if not step.is_setup:
                raise RuntimeError(
                    f"Step '{step.name}' was not setup!"
                )
            args = self._pack_args(args)
            args, config = step(*args, **kwargs, config=config)

        return args, config

    def _pack_args(self, args) -> Tuple:
        args = () if args is None else args
        if not isinstance(args, tuple):
            args = (args,)
        return args


def add_replace_field(
    config: Dict,
    key_chain: str,
    value: Any
) -> None:
    """Replace or add (if not present) a field in a dictionary, following a
    path of dot-separated keys. Inplace operation.

    Args:
        config (Dict): dictionary to be modified.
        key_chain (str): path of dot-separated keys to specify the location
        if the new value (e.g., 'foo.bar.line' adds/overwrites the value
        located at config['foo']['bar']['line']).
        value (Any): the value to insert.
    """
    sub_config = config
    for idx, k in enumerate(key_chain.split('.')):
        if idx >= len(key_chain.split('.')) - 1:
            # Last key reached
            break
        if not isinstance(sub_config.get(k), dict):
            sub_config[k] = dict()
        sub_config = sub_config[k]
    sub_config[k] = value


def load_pipeline_step(
    pipe: Union[str, Dict],
    step_id: Union[str, int],
    override_keys: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Executable:
    """Instantiates a specific step from a pipeline configuration file, given
    its ID (index if steps are a list, key if steps are a dictionary). It
    allows to override the step configuration with user defined values.

    Args:
        pipe (Union[str, Dict]): pipeline configuration. Either a path to a
        YAML file (if string), or a configuration in memory (if dict object).
        step_id (Union[str, int]): step identifier: list index if steps are
        represented as a list, string key if steps are represented as a
        dictionary.
        override_keys (Optional[Dict[str, Any]], optional): if given, maps key
        path to the value to add/override. A key path is a string of
        dot-separated keys (e.g., 'foo.bar.line' adds/overwrites the value
        located at pipe['foo']['bar']['line']). Defaults to None.
        verbose (bool, optional): if given, prints to console the new
        configuration, obtained after overriding. Defaults to False.

    Returns:
        Executable: an instance of the selected step in the pipeline. 
    """
    if isinstance(pipe, str):
        # Load pipe from YAML file path
        pipe = load_yaml(pipe)
    step_dict_config = pipe['executor']['init_args']['steps'][step_id]

    # Override fields
    if override_keys is not None:
        for key_chain, value in override_keys.items():
            add_replace_field(step_dict_config, key_chain, value)
    if verbose:
        import json
        print(f"NEW STEP <ID:{step_id}> CONFIG:")
        print(json.dumps(step_dict_config, indent=4))

    # Wrap config under "step" field and parse it
    step_dict_config = dict(step=step_dict_config)
    step_parser = ArgumentParser()
    step_parser.add_subclass_arguments(Executable, "step")
    parsed_namespace = step_parser.parse_object(step_dict_config)
    return step_parser.instantiate_classes(parsed_namespace)["step"]
