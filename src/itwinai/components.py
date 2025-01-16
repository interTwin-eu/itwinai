# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""This module provides the base classes to define modular and reproducible ML
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
using itwinai CLI without the need for python files.

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
components, thus complicating the definition of a structure structure beforehand
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

import functools
import time
from abc import ABC, abstractmethod

# import logging
# from logging import Logger as PythonLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .serialization import ModelLoader, Serializable
from .type import MLArtifact, MLDataset, MLModel


def monitor_exec(method: Callable) -> Callable:
    """Decorator for ``BaseComponent``'s methods.
    Prints when the component starts and ends executing, indicating
    its execution time.
    """

    @functools.wraps(method)
    def wrapper(self: BaseComponent, *args, **kwargs) -> Any:
        msg = f"Starting execution of '{self.name}'..."
        self._printout(msg)
        start_t = time.time()
        try:
            result = method(self, *args, **kwargs)
        finally:
            self.cleanup()
        self.exec_t = time.time() - start_t
        msg = f"'{self.name}' executed in {self.exec_t:.3f}s"
        self._printout(msg)
        return result

    return wrapper


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
    #: Dictionary storing constructor arguments. Needed to serialize the
    #: class to dictionary. Set by ``self.save_parameters()`` method.
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
        """Name of current component. Defaults to ``self.__class__.__name__``."""
        return self._name if self._name is not None else self.__class__.__name__

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @abstractmethod
    @monitor_exec
    def execute(self, *args, **kwargs) -> Any:
        """Execute some operations."""

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
        print("#" * len(msg))
        print(msg)
        print("#" * len(msg))


class DataGetter(BaseComponent):
    """Retrieves a dataset."""

    @abstractmethod
    @monitor_exec
    def execute(self) -> MLDataset:
        """Retrieves a dataset.

        Returns:
            MLDataset: retrieved dataset.
        """


class DataProcessor(BaseComponent):
    """Performs dataset pre-processing."""

    @abstractmethod
    @monitor_exec
    def execute(
        self,
        train_dataset: MLDataset,
        validation_dataset: MLDataset,
        test_dataset: MLDataset,
    ) -> Tuple[MLDataset, MLDataset, MLDataset]:
        """Trains a machine learning model.

        Args:
            train_dataset (MLDataset): training dataset.
            validation_dataset (MLDataset): validation dataset.
            test_dataset (MLDataset): test dataset.

        Returns:
            Tuple[MLDataset, MLDataset, MLDataset]: preprocessed training
            dataset, validation dataset, test dataset.
        """


class DataSplitter(BaseComponent):
    """Splits a dataset into train, validation, and test splits."""

    _train_proportion: Union[int, float]
    _validation_proportion: Union[int, float]
    _test_proportion: Union[int, float]

    def __init__(
        self,
        train_proportion: Union[int, float],
        validation_proportion: Union[int, float],
        test_proportion: Union[int, float],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))
        self.train_proportion = train_proportion
        self.validation_proportion = validation_proportion
        self.test_proportion = test_proportion

    @property
    def train_proportion(self) -> Union[int, float]:
        """Training set proportion."""
        return self._train_proportion

    @train_proportion.setter
    def train_proportion(self, prop: Union[int, float]) -> None:
        if isinstance(prop, float) and not 0.0 <= prop <= 1.0:
            raise ValueError(
                "Train proportion should be in the interval [0.0, 1.0] "
                f"if given as float. Received {prop}"
            )
        self._train_proportion = prop

    @property
    def validation_proportion(self) -> Union[int, float]:
        """Validation set proportion."""
        return self._validation_proportion

    @validation_proportion.setter
    def validation_proportion(self, prop: Union[int, float]) -> None:
        if isinstance(prop, float) and not 0.0 <= prop <= 1.0:
            raise ValueError(
                "Validation proportion should be in the interval [0.0, 1.0] "
                f"if given as float. Received {prop}"
            )
        self._validation_proportion = prop

    @property
    def test_proportion(self) -> Union[int, float]:
        """Test set proportion."""
        return self._test_proportion

    @test_proportion.setter
    def test_proportion(self, prop: Union[int, float]) -> None:
        if isinstance(prop, float) and not 0.0 <= prop <= 1.0:
            raise ValueError(
                "Test proportion should be in the interval [0.0, 1.0] "
                f"if given as float. Received {prop}"
            )
        self._test_proportion = prop

    @abstractmethod
    @monitor_exec
    def execute(self, dataset: MLDataset) -> Tuple[MLDataset, MLDataset, MLDataset]:
        """Splits a dataset into train, validation and test splits.

        Args:
            dataset (MLDataset): input dataset.

        Returns:
            Tuple[MLDataset, MLDataset, MLDataset]: tuple of
            train, validation and test splits.
        """


class Trainer(BaseComponent):
    """Trains a machine learning model."""

    @abstractmethod
    @monitor_exec
    def execute(
        self,
        train_dataset: MLDataset,
        validation_dataset: MLDataset,
        test_dataset: MLDataset,
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


class Predictor(BaseComponent):
    """Applies a pre-trained machine learning model to unseen data."""

    #: Pre-trained ML model used to make predictions.
    model: MLModel

    def __init__(
        self,
        model: Union[MLModel, ModelLoader],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.save_parameters(**self.locals2params(locals()))
        self.model = model() if isinstance(model, ModelLoader) else model

    @abstractmethod
    @monitor_exec
    def execute(
        self, predict_dataset: MLDataset, model: Optional[MLModel] = None
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


class Adapter(BaseComponent):
    """Connects to components in a sequential pipeline, allowing to
    control with greater detail how intermediate results are propagated
    among the components.

    Args:
        policy (List[Any]): list of the same length of the output of this
            component, describing how to map the input args to the output.
        name (Optional[str], optional): name of the component.
            Defaults to None.

    The adapter allows to define a policy with which inputs are re-arranged
    before being propagated to the next component.
    Some examples: [policy]: (input) -> (output)

    - ["INPUT_ARG#2", "INPUT_ARG#1", "INPUT_ARG#0"]: (11,22,33) -> (33,22,11)

    - ["INPUT_ARG#0", "INPUT_ARG#2", None]: (11, 22, 33) -> (11, 33, None)

    - []: (11, 22, 33) -> ()

    - [42, "INPUT_ARG#2", "hello"] -> (11,22,33,44,55) -> (42, 33, "hello")

    - [None, 33, 3.14]: () -> (None, 33, 3.14)

    - [None, 33, 3.14]: ("double", 44, None, True) -> (None, 33, 3.14)

    """

    #: Adapter policy.
    policy: List[Any]
    INPUT_PREFIX: str = "INPUT_ARG#"

    def __init__(self, policy: List[Any], name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.save_parameters(**self.locals2params(locals()))
        self.name = name
        self.policy = policy

    @monitor_exec
    def execute(self, *args) -> Tuple:
        """Produces an output tuple by arranging input arguments according
        to the policy specified in the constructor.

        Args:
            args (Tuple): input arguments.

        Returns:
            Tuple: input args arranged according to some policy.
        """
        result = []
        for itm in self.policy:
            if not (isinstance(itm, str) and itm.startswith(self.INPUT_PREFIX)):
                result.append(itm)
                continue

            arg_idx = int(itm[len(self.INPUT_PREFIX) :])
            if arg_idx >= len(args):
                max_idx = max(
                    map(
                        lambda itm: int(itm[len(self.INPUT_PREFIX) :]),
                        filter(
                            lambda el: (
                                isinstance(el, str) and el.startswith(self.INPUT_PREFIX)
                            ),
                            self.policy,
                        ),
                    )
                )
                raise IndexError(
                    f"The args received as input by '{self.name}' "
                    "are not consistent with the given adapter policy "
                    "because input args are too few! "
                    f"Input args are {len(args)} but the policy foresees "
                    f"at least {max_idx+1} items."
                )
            result.append(args[arg_idx])
        return tuple(result)
