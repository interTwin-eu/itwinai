"""
This module provides the base classes to define modular and reproducible ML
workflows. The base component classes provide a template to follow for
extending existing components or creating new ones.

There are two ways of creating workflows: simple and advanced workflows.

Simple workflows can be obtained by creating a sequence of components
wrapped in a Pipeline object, which executes them in cascade, passing the
output of a component as the input of the following one. It is responsibility
of the user to prevent mismatches among outputs and inputs of component
sequences. This pipeline can be configured
both in terms of parameters and structure, with a configuration file
representing the whole pipeline. This configuration file can be executed
using itwinai CLI without the need of python files.

Example:

>>> from itwinai.components import DataGetter, Saver
>>> from itwinai.pipeline import Pipeline
>>>
>>> my_pipe = Pipeline({"getter": DataGetter(...), "data_saver": Saver(...)})
>>> my_pipe.execute()
>>> my_pipe.to_yaml("training_pipe.yaml")
>>>
>>> # The pipeline can be parsed back to Python with:
>>> from itwinai.parser import PipeParser
>>> my_pipe = PipeParser("training_pipe.yaml")
>>> my_pipe.execute()
>>>
>>> # Run the pipeline from configuration file with dynamic override
>>> itwinai exec-pipeline --config training_pipe.yaml \
>>> --override pipeline.init_args.steps.data_saver.some_param 42


Advanced workflows foresee more complicated connections between the
components and it is very difficult to define a structure beforehand
without risking of over-constraining the user. Therefore, advanced
workflows are defined by explicitly connecting component outputs to
to the inputs of other components, without a wrapper Pipeline object.
In this case, the configuration files enable the user to persist the
parameters passed to the argument parser, enabling reuse through
configuration files, with the possibility of dynamic overrides of parameters.

Example:

>>> from jsonargparse import ArgumentParser, ActionConfigFile
>>>
>>> parser = ArgumentParser(description='PyTorch MNIST Example')
>>> parser.add_argument('--batch-size', type=int, default=64,
>>>                     help='input batch size for training (default: 64)')
>>> parser.add_argument('--epochs', type=int, default=10,
>>>                     help='number of epochs to train (default: 10)')
>>> parser.add_argument('--lr', type=float, default=0.01,
>>>                     help='learning rate (default: 0.01)')
>>> parser.add_argument(
>>>     "-c", "--config", action=ActionConfigFile,
>>>     required=True,
>>>     help="Path to a configuration file in json or yaml format."
>>> )
>>> args = parser.parse_args()
>>>
>>> from itwinai.components import (
>>>     DataGetter, Saver, DataSplitter, Trainer
>>> )
>>> getter = DataGetter(...)
>>> splitter = DataSplitter(...)
>>> data_saver = Saver(...)
>>> model_saver = Saver(...)
>>> trainer = Trainer(
>>>     batch_size=args.batch_size, lr=args.lr, epochs=args.epochs
>>> )
>>>
>>> # Compose workflow
>>> my_dataset = getter.execute()
>>> train_set, valid_set, test_set = splitter.execute(my_dataset)
>>> data_saver.execute("train_dataset.pkl", test_set)
>>> _, _, _, trained_model = trainer(train_set, valid_set)
>>> model_saver.execute(trained_model)
>>>
>>> # Run the script using a previous configuration with dynamic override
>>> python my_train.py --config training_pipe.yaml --lr 0.002
"""


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



def monitor_exec(method: Callable) -> Callable:
    """Decorator for execute method of a component class.
    Computes execution time and gives some information about
    the execution of the component.

    Args:
        func (Callable): class method.
    """
    @functools.wraps(method)
    def monitored_method(self: BaseComponent, *args, **kwargs) -> Any:
        msg = f"Starting execution of '{self.name}'..."
        self._printout(msg)
        start_t = time.time()
        try:
            # print(f'ARGS: {args}')
            # print(f'KWARGS: {kwargs}')
            result = method(self, *args, **kwargs)
        finally:
            self.cleanup()
        self.exec_t = time.time() - start_t
        msg = f"'{self.name}' executed in {self.exec_t:.3f}s"
        self._printout(msg)
        return result

    return monitored_method


class BaseComponent(ABC, Serializable):
    """Base component class. Each component provides a simple interface
    to foster modularity in machine learning code. Each component class
    implements the `execute` method, which received some input ML artifacts
    (e.g., datasets), performs some operations and returns new artifacts.
    The components are meant to be assembled in complex ML workflows,
    represented as pipelines.

        Args:
            name (Optional[str], optional): unique identifier for a step.
                Defaults to None.
        """
    _name: str = None
    parameters: Dict[Any, Any] = None

    def __init__(
        self,
        name: Optional[str] = None,
        # logs_dir: Optional[str] = None,
        # debug: bool = False,
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))
        self.name = name

    @property
    def name(self) -> str:
        return (
            self._name if self._name is not None else self.__class__.__name__
        )

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @abstractmethod
    @monitor_exec
    def execute(self, *args, **kwargs) -> Any:
        """"Execute some operations."""

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

    def cleanup(self):
        """Cleanup resources allocated by this component."""

    @staticmethod
    def _printout(msg: str):
        msg = f"# {msg} #"
        print("#"*len(msg))
        print(msg)
        print("#"*len(msg))


class Trainer(Executable):
    """Trains a machine learning model."""
    @abstractmethod
    @monitor_exec
    def execute(
        self,
        train_dataset: MLDataset,
        validation_dataset: MLDataset,
        test_dataset: MLDataset
    ) -> Tuple[MLDataset, MLDataset, MLDataset, MLModel]:
        """Trains a machine learning model.

        Args:
            train_dataset (MLDataset): training dataset.
            validation_dataset (MLDataset): validation dataset.
            test_dataset (MLDataset): test dataset.

        Returns:
            Tuple[MLDataset, MLDataset, MLDataset]: training dataset,
            validation dataset, test dataset, trained model.
        """

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

    @abstractmethod
    @monitor_exec
    def execute(
        self,
        predict_dataset: MLDataset,
        model: Optional[MLModel] = None
    ) -> MLDataset:
        """Applies a machine learning model on a dataset of samples.

        Args:
            predict_dataset (MLDataset): dataset for inference.
            model (Optional[MLModel], optional): overrides the internal model,
                if given. Defaults to None.

        Returns:
            MLDataset: predictions with the same cardinality of the
                input dataset.
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
