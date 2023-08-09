import yaml
import ray

from .components import Executor
from jsonargparse import ArgumentParser
from itwinai.backend.utils import parse_pipe_config
from ray import air, tune


class LocalExecutor(Executor):
    def __init__(self, pipeline, class_dict):
        # Create parser for the pipeline (ordered)
        pipe_parser = ArgumentParser()
        for k, v in class_dict.items():
            pipe_parser.add_subclass_arguments(v, k)

        # Parse, Instantiate pipe
        if type(pipeline) == str:
            parsed = parse_pipe_config(pipeline, pipe_parser)
        elif type(pipeline) == dict:
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
        self.pipeline = pipeline
        self.class_dict = class_dict
        self.param_space = param_space

        # Init ray
        ray.init()

    def worker_fn(self, config, pipeline, class_dict):
        print('Worker fn')
        # Should have same structure pipe and params
        # def replace(pipe, params):
        #     for param in params:
        #         if type(pipe[param]) != dict:
        #             pipe[param] = params[param]
        #         else:
        #             replace(pipe[param], params[param])
        #     return pipe
        #
        # template = pipeline
        # with open(template, 'r') as f:
        #     doc = yaml.safe_load(f)
        # doc = replace(doc, config)
        #
        # executor = LocalExecutor(doc, class_dict)
        # executor.setup(None)
        # executor.execute(None)
        return {"test": 1}

    def execute(self, args):
        print('Execute')
        tuner = tune.Tuner(
            trainable=tune.with_parameters(self.worker_fn, pipeline=self.pipeline, class_dict=self.class_dict),
            param_space=self.param_space,
            run_config=air.RunConfig(name="tune_run")
        )
        results = tuner.fit()
        print(f"Best hyperparameters found were: {results.get_best_result().config}")

    # Setup is done per worker via Tune execution
    def setup(self, args):
        pass