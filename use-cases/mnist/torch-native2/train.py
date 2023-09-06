"""
Training pipeline. To run this script, use the following command:

>>> torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29400 train.py -p pipeline.yaml

Or:

>>> torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29400 use-cases/mnist/torch-native/train.py \
        -p use-cases/mnist/torch-native/pipeline.yaml
"""

import argparse

from itwinai.backend.components import Executor
from itwinai.backend.utils import parse_pipe_config
from jsonargparse import ArgumentParser


if __name__ == "__main__":
    # Create CLI Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pipeline", type=str, required=True)
    args = parser.parse_args()

    # Create parser for the pipeline (ordered)
    pipe_parser = ArgumentParser()
    pipe_parser.add_subclass_arguments(Executor, "executor")

    # Parse, Instantiate pipe
    parsed = parse_pipe_config(args.pipeline, pipe_parser)
    pipe = pipe_parser.instantiate_classes(parsed)
    executor = getattr(pipe, 'executor')
    executor.setup()
    executor.execute()
