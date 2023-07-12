from ..components import Executor


class TensorflowExecutor(Executor):
    def __init__(self):
        pass

    def execute(self, pipeline):
        args = None
        for executable in pipeline:
            args = executable.execute(args)