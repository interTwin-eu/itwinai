from trainer import TensorflowTrainer
from dataloader import TensorflowDataGetter, TensorflowDataPreproc
from itwinai.backend.tensorflow.executor import TensorflowExecutor
from itwinai.backend.tensorflow.utils import parse_pipe_config
from jsonargparse import ArgumentParser


if __name__ == '__main__':
    # Create parser for the pipeline (ordered)
    parser = ArgumentParser()
    parser.add_subclass_arguments(TensorflowDataGetter, 'getter')
    parser.add_subclass_arguments(TensorflowDataPreproc, 'preproc')
    parser.add_subclass_arguments(TensorflowTrainer, 'trainer')

    # Parse, Instantiate pipe
    parsed = parse_pipe_config('pipeline.yaml', parser)
    pipe = parser.instantiate_classes(parsed)
    # Make pipe as a list
    pipe = [getattr(pipe, arg) for arg in vars(pipe)]

    # Execute pipe
    executor = TensorflowExecutor()
    executor.execute(pipe)