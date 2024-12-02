#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# DISCLAIMER: 
# this script is here to support the development, so it may not be maintained and it may be a bit "rough".
# Do not mind it too much.

# Python virtual environment
PYTHON_VENV="../../.venv-pytorch"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun logs_mpirun logs_srun

# Adapt with a path reachable by you
export MNIST_PATH="/ceph/hpc/data/st2301-itwin-users/mbunino/mnist" #"/p/project1/intertwin/smalldata/mnist"

# Containers
# - itwinai_torch.sif: cmcc jlab container (OMPI v5)
# - itwinai_torch2.sif: itwinai 0.2.2.dev torch2.4 (OMPI v4.1)
# - itwinai_torch3.sif: itwinai 0.2.2.dev2 torch2.4 - force distributed (OMPI v4.1)
# - itwinai_torch4.sif: cmcc jlab container (OMPI v4.1)
# - /ceph/hpc/data/st2301-itwin-users/mbunino/jlab_simple_reconstructed_nv_itwinai.sif: jlab container recostructed from simple (OMPI v4.1)
export CONTAINER_PATH="container1.sif"

# Disable pytest ANSI coloring
export NO_COLOR=1

export DIST_MODE="ddp"
export RUN_NAME="ddp-itwinai"
export COMMAND='pytest -v -m torch_dist /app/tests'
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh

export DIST_MODE="deepspeed"
export RUN_NAME="ds-itwinai"
export COMMAND='pytest -v -m deepspeed_dist /app/tests'
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh

export DIST_MODE="horovod"
export RUN_NAME="horovod-itwinai"
export COMMAND="pytest -v -m horovod_dist /app/tests"
sbatch \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh
