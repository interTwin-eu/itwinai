from abc import ABCMeta, abstractmethod

class Executable(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, args):
        pass

    @abstractmethod
    def config(self, config):
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

class Executor(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, pipeline):
        pass

class Logger(metaclass=ABCMeta):
    @abstractmethod
    def log(self):
        pass