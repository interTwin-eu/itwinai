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
from typing import Any, Optional, Tuple, Union, Callable, Dict
from abc import ABC, abstractmethod
import time
import functools
# import logging
# from logging import Logger as PythonLogger

from .types import MLModel, MLDataset, MLArtifact
from .serialization import ModelLoader, Serializable


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
    _name: str = 'unnamed'
    parameters: Dict[Any, Any] = None

    def __init__(
        self,
        name: Optional[str] = None,
        # logs_dir: Optional[str] = None,
        # debug: bool = False,
    ) -> None:
        self.save_parameters(name=name)

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


class Trainer(BaseComponent):
    """Trains a machine learning model."""

    @abstractmethod
    @monitor_exec
    def execute(
        self,
        train_dataset: MLDataset,
        validation_dataset: MLDataset
    ) -> Tuple[MLDataset, MLDataset, MLModel]:
        """Trains a machine learning model.

        Args:
            train_dataset (DatasetML): training dataset.
            validation_dataset (DatasetML): validation dataset.

        Returns:
            Tuple[DatasetML, DatasetML, ModelML]: training dataset,
            validation dataset, trained model.
        """

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass


class Predictor(BaseComponent):
    """Applies a pre-trained machine learning model to unseen data."""

    model: MLModel

    def __init__(
        self,
        model: Union[MLModel, ModelLoader],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.save_parameters(model=model, name=name)
        self.model = model() if isinstance(model, ModelLoader) else model

    @abstractmethod
    @monitor_exec
    def execute(
        self,
        predict_dataset: MLDataset,
        model: Optional[MLModel] = None
    ) -> MLDataset:
        """Applies a machine learning model on a dataset of samples.

        Args:
            predict_dataset (DatasetML): dataset for inference.
            model (Optional[ModelML], optional): overrides the internal model,
                if given. Defaults to None.

        Returns:
            DatasetML: predictions with the same cardinality of the
                input dataset.
        """


class DataGetter(BaseComponent):
    """Retrieves a dataset."""

    @abstractmethod
    @monitor_exec
    def execute(self) -> MLDataset:
        """Retrieves a dataset.

        Returns:
            MLDataset: retrieved dataset.
        """


class DataPreproc(BaseComponent):
    """Performs dataset pre-processing."""

    @abstractmethod
    @monitor_exec
    def execute(self, dataset: MLDataset) -> MLDataset:
        """Pre-processes a dataset.

        Args:
            dataset (MLDataset): dataset.

        Returns:
            MLDataset: pre-processed dataset.
        """


class Saver(BaseComponent):
    """Saves artifact to disk."""

    @abstractmethod
    @monitor_exec
    def execute(self, artifact: MLArtifact) -> MLArtifact:
        """Saves an ML artifact to disk.

        Args:
            artifact (MLArtifact): artifact to save.

        Returns:
            MLArtifact: the same input artifact, after saving it.
        """
