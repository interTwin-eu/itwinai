# TODO: Solve relative import
import sys
sys.path.append("..")
from components import Executor


class TensorflowExecutor(Executor):
    def __init__(self):
        pass

    def execute(self, pipeline):
        args = None
        for executable in pipeline:
            args = executable.execute(args)

    def config(self, pipeline, config):
        for configurable in pipeline:
            print(configurable)
            configurable.config(config)