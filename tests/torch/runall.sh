#!/bin/bash

# Python virtual environment
PYTHON_VENV="../../.venv-pytorch"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun logs_mpirun

export MNIST_PATH="/ceph/hpc/data/st2301-itwin-users/mbunino/mnist" #"/p/project1/intertwin/smalldata/mnist"

# Containers
# - itwinai_torch.sif: cmcc jlab container
# - itwinai_torch2.sif: itwinai 0.2.2.dev torch2.4
# - itwinai_torch3.sif: itwinai 0.2.2.dev2 torch2.4 - force distributed
export CONTAINER_PATH="itwinai_torch3.sif"

# Disable pytest ANSI coloring
export NO_COLOR=1

DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD='pytest -vs -m torch_dist' # test_distribtued.py' # test_torch_trainer.py'
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh

DIST_MODE="deepspeed"
RUN_NAME="ds-itwinai"
TRAINING_CMD='pytest -vs -m deepspeed_dist' # test_distribtued.py' # test_torch_trainer.py'
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh

DIST_MODE="horovod"
RUN_NAME="horovod-itwinai"
TRAINING_CMD="pytest -vs -m horovod_dist"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh