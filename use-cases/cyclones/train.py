# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Roman Machacek
#
# Credit:
# - Roman Machacek <roman.machacek@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""
Training pipeline. To run this script, use the following commands.

On login node:

>>> python train.py -p pipeline.yaml -d

On compute nodes:

>>>  python train.py -p pipeline.yaml

"""

from typing import Dict
import argparse
import logging
from os.path import join
from os import makedirs
from datetime import datetime

# # the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
from itwinai.parser import ConfigParser, ArgumentParser

from src.macros import PATCH_SIZE, SHAPE


def dynamic_config(args) -> Dict:
    """Generates a configuration with values computed at runtime.

    Args:
        args (argparse.Namespace): arguments parsed from command line.

    Returns:
        Dict: configuration.
    """
    config = {}

    # Paths, Folders
    FORMATTED_DATETIME = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    MODEL_BACKUP_DIR = join(args.root_dir, "models/")
    EXPERIMENTS_DIR = join(args.root_dir, "experiments")
    RUN_DIR = join(EXPERIMENTS_DIR, args.run_name + "_" + FORMATTED_DATETIME)
    SCALER_DIR = join(RUN_DIR, "scalers")
    TENSORBOARD_DIR = join(RUN_DIR, "tensorboard")
    CHECKPOINTS_DIR = join(RUN_DIR, "checkpoints")

    # Files
    LOG_FILE = join(RUN_DIR, "run.log")

    # Create folders
    makedirs(EXPERIMENTS_DIR, exist_ok=True)
    makedirs(RUN_DIR, exist_ok=True)
    makedirs(SCALER_DIR, exist_ok=True)
    makedirs(TENSORBOARD_DIR, exist_ok=True)
    makedirs(CHECKPOINTS_DIR, exist_ok=True)

    config = {
        "root_dir": args.root_dir,
        "experiment_dir": EXPERIMENTS_DIR,
        "run_dir": RUN_DIR,
        "scaler_dir": SCALER_DIR,
        "tensorboard_dir": TENSORBOARD_DIR,
        "checkpoints_dir": CHECKPOINTS_DIR,
        "backup_dir": MODEL_BACKUP_DIR,
        "log_file": LOG_FILE,
        "shape": SHAPE,
        "patch_size": PATCH_SIZE,
        # "epochs": args.epochs,
        # "batch_size": args.batch_size
    }

    # initialize logger
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s : %(message)s",
        level=logging.DEBUG,
        filename=LOG_FILE,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--pipeline",
        type=str,
        required=True,
        help="Configuration file to the pipeline to execute.",
    )
    parser.add_argument("-r", "--root_dir", type=str, default="./data")
    parser.add_argument("--data_path", type=str, default="./data/data_path")
    parser.add_argument("-n", "--run_name", default="noname", type=str)
    parser.add_argument(
        "-d",
        "--download-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to download only the dataset and exit execution "
            "(suggested on login nodes of HPC systems)"
        ),
    )

    args = parser.parse_args()
    global_config = dynamic_config(args)

    # Create parser for the pipeline
    pipe_parser = ConfigParser(
        config=args.pipeline,
        override_keys={"dataset_root": args.data_path, "global_config": global_config},
    )
    pipeline = pipe_parser.parse_pipeline(pipeline_nested_key="training_pipeline")

    if args.download_only:
        print("Downloading datasets and exiting...")
        pipeline = pipeline[:1]

    pipeline.execute()
