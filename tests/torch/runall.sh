#!/bin/bash

# Python virtual environment
PYTHON_VENV=".venv-docs"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun

# Torchrun-based tests for HPC for itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD='pytest -vs -m torchrun' # test_torch_trainer.py'
export MNIST_PATH="/p/project1/intertwin/smalldata/mnist"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh

# # Horovodrun-based tests for HPC for itwinai
# DIST_MODE="horovod"
# RUN_NAME="horovod-itwinai"
# TRAINING_CMD="pytest -vs -m mpirun test_torch_trainer.py"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.sh