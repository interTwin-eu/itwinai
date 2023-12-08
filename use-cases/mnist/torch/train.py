"""
Training pipeline. To run this script, use the following commands.

On login node:

>>> micromamba run -p ../../../.venv-pytorch/ \
    python train.py -p pipeline.yaml -d

On compute nodes:

>>> micromamba run -p ../../../.venv-pytorch/ \
    python train.py -p pipeline.yaml

"""

import argparse

from itwinai.components import Pipeline
from itwinai.utils import parse_pipe_config
from jsonargparse import ArgumentParser


if __name__ == "__main__":
    # Create CLI Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pipeline", type=str, required=True,
        help='Configuration file to the pipeline to execute.'
    )
    parser.add_argument(
        '-d', '--download-only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=('Whether to download only the dataset and exit execution '
              '(suggested on login nodes of HPC systems)')
    )
    args = parser.parse_args()

    # Create parser for the pipeline (ordered)
    pipe_parser = ArgumentParser()
    pipe_parser.add_subclass_arguments(Pipeline, "executor")

    # Parse, Instantiate pipe
    parsed = parse_pipe_config(args.pipeline, pipe_parser)
    pipe = pipe_parser.instantiate_classes(parsed)
    executor: Pipeline = getattr(pipe, 'executor')

    if args.download_only:
        print('Downloading datasets and exiting...')
        executor = executor[:1]
    else:
        print('Downloading datasets (if not already done) and running...')
        executor = executor
    executor.setup()
    executor()
