"""Executors to execute a sequence of executable steps."""

from typing import Any, Dict, Iterable
from abc import abstractmethod

from itwinai.backend.components import Executable
from .components import Executor


class ParallelExecutor(Executor):
    """Execute a pipeline in parallel: multiprocessing and multi-node."""

    def __init__(self, steps: Iterable[Executable]):
        super().__init__(steps)

    def setup(self, config: Dict = None):
        return super().setup(config)

    def execute(self, args: Any = None):
        return super().execute(args)


class HPCExecutor(ParallelExecutor):
    """Execute a pipeline on an HPC system.
    This executor provides as additional `setup_on_login` method
    to allow for specific setup operations to be carried out on
    the login node of a GPU cluster, being the only one with
    network access.
    """

    def __init__(self, steps: Iterable[Executable]):
        super().__init__(steps)

    def setup(self, config: Dict = None):
        return super().setup(config)

    @abstractmethod
    def setup_on_login(self, config: Dict = None):
        """Access the network to download datasets and misc."""
        raise NotImplementedError

    def execute(self, args: Any = None):
        return super().execute(args)
