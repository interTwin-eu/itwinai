import argparse

from jsonargparse import ArgumentParser
from trainer import TensorflowTrainer
from dataloader import TensorflowDataGetter
from executor import CycloneExecutor
from itwinai.backend.utils import parse_pipe_config

if __name__ == "__main__":
    # Create CLI Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline", type=str)
    parser.add_argument("-r", "--root_dir", type=str)
    args = parser.parse_args()

    print(args.root_dir)

    # Create parser for the pipeline (ordered)
    pipe_parser = ArgumentParser()
    pipe_parser.add_subclass_arguments(CycloneExecutor, "executor")
    pipe_parser.add_subclass_arguments(TensorflowDataGetter, "getter")
    pipe_parser.add_subclass_arguments(TensorflowTrainer, "trainer")

    # Parse, Instantiate pipe
    parsed = parse_pipe_config(args.pipeline, pipe_parser)
    steps = pipe_parser.instantiate_classes(parsed)
    # Extract Executor, Steps in Pipe
    executor = steps.executor
    pipe = [getattr(steps, arg) for arg in list(vars(steps))[1:]]

    # Run pipe
    executor.setup([pipe, args.root_dir])
    executor.execute(pipe)
