"""
This module provides the base classes to define modular and reproducible ML
workflows. The base component classes provide a template to follow for
extending existing components or creating new ones.
"""


from typing import Any, Optional, Tuple, Union, Callable
from abc import ABCMeta, abstractmethod
import time
import functools
# import logging
# from logging import Logger as PythonLogger

from .types import MLModel, MLDataset, MLArtifact
from .serialization import ModelLoader


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


class BaseComponent(metaclass=ABCMeta):
    """Base component class.

        Args:
            name (Optional[str], optional): unique identifier for a step.
                Defaults to None.
        """
    name: str = 'unnamed'

    def __init__(
        self,
        name: Optional[str] = None,
        # logs_dir: Optional[str] = None,
        # debug: bool = False,
        **kwargs
    ) -> None:
        self.name = name if name is not None else self.__class__.__name__

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
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
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
