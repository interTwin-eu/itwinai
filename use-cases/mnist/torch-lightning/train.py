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

from itwinai.parser import ConfigParser


if __name__ == "__main__":
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

    # Create parser for the pipeline
    pipe_parser = ConfigParser(config=args.pipeline)
    pipeline = pipe_parser.parse_pipeline()

    if args.download_only:
        print('Downloading datasets and exiting...')
        pipeline = pipeline[:1]

    pipeline.execute()
