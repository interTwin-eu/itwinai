from ..components import Executor


class TensorflowExecutor(Executor):
    def __init__(self, args):
        self.args = args

    def execute(self, pipeline):
        args = None
        for executable in pipeline:
            args = executable.execute(args)

    def setup(self, pipeline):
        for executable in pipeline:
            executable.setup(self.args)