import argparse

from trainer import TensorflowTrainer
from dataloader import TensorflowDataGetter, TensorflowDataPreproc
from itwinai.backend.tensorflow.executor import TensorflowExecutor
from itwinai.backend.utils import parse_pipe_config
from jsonargparse import ArgumentParser


if __name__ == "__main__":
    # Create CLI Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline", type=str)
    args = parser.parse_args()

    # Create parser for the pipeline (ordered)
    pipe_parser = ArgumentParser()
    pipe_parser.add_subclass_arguments(TensorflowDataGetter, "getter")
    pipe_parser.add_subclass_arguments(TensorflowDataPreproc, "preproc")
    pipe_parser.add_subclass_arguments(TensorflowTrainer, "trainer")

    # Parse, Instantiate pipe
    parsed = parse_pipe_config(args.pipeline, pipe_parser)
    pipe = pipe_parser.instantiate_classes(parsed)
    # Make pipe as a list
    pipe = [getattr(pipe, arg) for arg in vars(pipe)]

    # Execute pipe
    executor = TensorflowExecutor(args={})
    executor.setup(pipe)
    executor.execute(pipe)
