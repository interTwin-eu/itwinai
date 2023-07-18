from ..components import Executor


class TorchExecutor(Executor):
    def __init__(self):
        pass

    def execute(self, pipeline):
        args = None
        for executable in pipeline:
            args = executable.execute(args)

    def setup(self, pipeline):
        args = None
        for executable in pipeline:
            args = executable.setup(args)
