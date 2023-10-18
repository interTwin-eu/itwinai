"""Executors to execute a sequence of executable steps."""

from typing import Any, Dict, Iterable
from abc import abstractmethod

import yaml
import ray
from ray import air, tune
from jsonargparse import ArgumentParser

from .components import Executor, Executable
from .utils import parse_pipe_config


class LocalExecutor(Executor):
    def __init__(self, pipeline, class_dict):
        # Create parser for the pipeline (ordered)
        pipe_parser = ArgumentParser()
        for k, v in class_dict.items():
            pipe_parser.add_subclass_arguments(v, k)

        # Parse, Instantiate pipe
        if isinstance(pipeline, str):
            parsed = parse_pipe_config(pipeline, pipe_parser)
        elif isinstance(pipeline, dict):
            parsed = pipe_parser.parse_object(pipeline)
        else:
            raise "Type of pipeline is not supported"

        pipe = pipe_parser.instantiate_classes(parsed)
        # Make pipe as a list
        self.pipe = [getattr(pipe, arg) for arg in vars(pipe)]

    def execute(self, args):
        for executable in self.pipe:
            args = executable.execute(args)

    def setup(self, args):
        for executable in self.pipe:
            args = executable.setup(args)


class RayExecutor(Executor):
    def __init__(self, pipeline, class_dict, param_space):
        self.class_dict = class_dict
        self.param_space = param_space

        # Read pipeline as yaml
        with open(pipeline, 'r') as f:
            self.pipeline = yaml.safe_load(f)

        # Init ray
        ray.init(ignore_reinit_error=True)
        print('Ray is initialized')

    def worker_fn(self, config, pipeline, class_dict):
        # Should have same structure pipe and params
        def replace(pipe, params):
            for param in params:
                if not isinstance(pipe[param], dict):
                    pipe[param] = params[param]
                else:
                    replace(pipe[param], params[param])
            return pipe

        doc = replace(pipeline, config)

        executor = LocalExecutor(doc, class_dict)
        executor.setup(None)
        executor.execute(None)

    def execute(self, args):
        print('Execute')
        tuner = tune.Tuner(
            trainable=tune.with_parameters(
                self.worker_fn,
                pipeline=self.pipeline,
                class_dict=self.class_dict
            ),
            param_space=self.param_space,
            run_config=air.RunConfig(name="tune_run")
        )
        results = tuner.fit()
        print(
            "Best hyperparameters found were: "
            f"{results.get_best_result().config}"
        )

    # Setup is done per worker via Tune execution
    def setup(self, args):
        pass


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
