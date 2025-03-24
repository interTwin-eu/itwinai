#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# shellcheck disable=all

# DISCLAIMER: 
# this script is here to support the development, so it may not be maintained and it may be a bit "rough".
# Do not mind it too much.

# Python virtual environment
export PYTHON_VENV="../../.venv"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun logs_mpirun logs_srun checkpoints

# Adapt with a path reachable by you
export MNIST_PATH="/p/project1/intertwin/smalldata/mnist"


# export CONTAINER_PATH="container1.sif"

# Disable pytest ANSI coloring
export NO_COLOR=1

export DIST_MODE="ray"
export RUN_NAME="ray-itwinai"
export COMMAND='pytest -v -m ray_dist test_tuning.py'
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.jsc.sh

# export DIST_MODE="ddp"
# export RUN_NAME="ddp-itwinai"
# export COMMAND='pytest -v -m torch_dist '
# sbatch  \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.jsc.sh

# export DIST_MODE="deepspeed"
# export RUN_NAME="ds-itwinai"
# export COMMAND='pytest -v -m deepspeed_dist '
# sbatch  \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.jsc.sh

# export DIST_MODE="horovod"
# export RUN_NAME="horovod-itwinai"
# export COMMAND="pytest -v -m horovod_dist "
# sbatch \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.jsc.sh
