from __future__ import annotations
from typing import Iterable, Dict, Any

from abc import ABCMeta, abstractmethod


class Executable(metaclass=ABCMeta):
    _is_setup: bool = False

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.execute(*args, **kwds)

    @abstractmethod
    def execute(self, args):
        pass

    @abstractmethod
    def setup(self, args: Dict):
        pass

    def cleanup(self, args):
        pass


class Trainer(Executable):
    @abstractmethod
    def train(self, data):
        pass


class DataGetter(Executable):
    @abstractmethod
    def load(self, args):
        pass


class DataPreproc(Executable):
    @abstractmethod
    def preproc(self, args):
        pass


class StatGetter(Executable):
    @abstractmethod
    def stats(self, args):
        pass


class Evaluator(Executable):
    @abstractmethod
    def evaluate(self, args):
        pass


class Saver(Executable):
    @abstractmethod
    def save(self, args):
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
        super().__init__()
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

    def execute(self, args: Any = None):
        """Execute the pipeline step-by-step providing as input
        to the next step the output of the previous one.

        Args:
            args (Any, optional): Input of the first step of the pipeline.
                Defaults to None.
        """
        for step in self.steps:
            args = step.execute(args)


class Logger(metaclass=ABCMeta):
    savedir: str = None

    @abstractmethod
    def log(self, args):
        pass
