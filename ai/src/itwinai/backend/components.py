from __future__ import annotations
from typing import Iterable, Dict, Any, Optional, Tuple

from abc import ABCMeta, abstractmethod
import time

from ..cluster import ClusterEnvironment


class Executable(metaclass=ABCMeta):
    name: str = 'unnamed'
    _is_setup: bool = False
    cluster: ClusterEnvironment

    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        self.name = name if name is not None else self.__class__.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # This method SHOULD NOT be overridden. This is just a wrapper.
        # Override execute() instead!
        msg = f"Starting execution of '{self.name}'..."
        self._printout(msg)
        start_t = time.time()
        try:
            result = self.execute(*args, **kwargs)
        finally:
            self.cleanup()
        self.exec_t = time.time() - start_t
        msg = f"'{self.name}' executed in {self.exec_t:.3f}s"
        self._printout(msg)
        return result

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self, config: Dict):
        pass

    def cleanup(self):
        pass

    def _printout(self, msg: str):
        msg = f"# {msg} #"
        print("\n" + "#"*len(msg))
        print(msg)
        print("#"*len(msg) + "\n")


class Trainer(Executable):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass


class DataGetter(Executable):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass


class DataPreproc(Executable):
    @abstractmethod
    def preproc(self, *args, **kwargs):
        pass


class StatGetter(Executable):
    @abstractmethod
    def stats(self, *args, **kwargs):
        pass


class Evaluator(Executable):
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


class Saver(Executable):
    @abstractmethod
    def save(self, *args, **kwargs):
        pass


class Executor(Executable):
    """Sets-up and executes a sequence of Executable steps."""

    steps: Iterable[Executable]
    constructor_args: Dict

    def __init__(
        self,
        steps: Iterable[Executable],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.constructor_args = kwargs

    def __getitem__(self, subscript) -> Executor:
        if isinstance(subscript, slice):
            s = self.steps[subscript.start:subscript.stop: subscript.step]
            sliced = self.__class__(
                steps=s,
                **self.constructor_args
            )
            return sliced
        else:
            return self.steps[subscript]

    def __len__(self) -> int:
        return len(self.steps)

    def setup(self, config: Dict = None):
        """Pass a key-value based configuration down the pipeline,
        to propagate information computed at real-time.

        Args:
            config (Dict, optional): key-value configuration. Defaults to None.
        """
        for step in self.steps:
            config = step.setup(config)

    def execute(self, args: Optional[Tuple] = None):
        """Execute the pipeline step-by-step providing as input
        to the next step the output of the previous one.

        Args:
            args (Any, optional): Input of the first step of the pipeline.
                Defaults to None.
        """
        for step in self.steps:
            args = () if args is None else args
            args = step(*args)


class Logger(metaclass=ABCMeta):
    savedir: str = None

    @abstractmethod
    def log(self, args):
        pass
