import argparse

from trainer import MNISTTrainer
from dataloader import MNISTDataGetter, MNISTDataPreproc
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
    pipe_parser.add_subclass_arguments(MNISTDataGetter, "getter")
    pipe_parser.add_subclass_arguments(MNISTDataPreproc, "preproc")
    pipe_parser.add_subclass_arguments(MNISTTrainer, "trainer")

    # Parse, Instantiate pipe
    parsed = parse_pipe_config(args.pipeline, pipe_parser)
    pipe = pipe_parser.instantiate_classes(parsed)
    # Make pipe as a list
    pipe = [getattr(pipe, arg) for arg in vars(pipe)]

    # Execute pipe
    executor = TensorflowExecutor(args={})
    executor.setup(pipe)
    executor.execute(pipe)
