from typing import Iterable, Dict

from abc import ABCMeta, abstractmethod


class Executable(metaclass=ABCMeta):
    _is_setup: bool = False

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
    def __init__(
        self,
        steps: Iterable[Executable],
    ):
        super().__init__()
        self.steps = steps


class Logger(metaclass=ABCMeta):
    savedir: str = None

    @abstractmethod
    def log(self, args):
        pass
