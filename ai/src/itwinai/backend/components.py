from abc import ABCMeta, abstractmethod


class Executable(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, args):
        pass

    @abstractmethod
    def setup(self, args):
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
    @abstractmethod
    def execute(self, pipeline):
        pass

    @abstractmethod
    def setup(self, pipeline):
        pass


class Logger(metaclass=ABCMeta):
    savedir: str = None

    @abstractmethod
    def log(self, args):
        pass
