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

# Select on which system you want to run the tests
WHERE_TO_RUN=$1 #"jsc" # "vega"

# Global config
NUM_NODES=2
GPUS_PER_NODE=2

# HPC-wise config
if [[ $WHERE_TO_RUN == "jsc" ]]; then

    # JSC (venv)
    export MNIST_PATH="/p/project1/intertwin/smalldata/mnist"
    export PYTHON_VENV="../../.venv"
    # Path to shared filesystem that all the Ray workers can access. /tmp is a local filesystem path to each worker
    # This is only needed by tests. This is needed by the torch trainer to store and retrieve checkpoints
    export SHARED_FS_PATH="/p/project1/intertwin/bunino1/tmp"

    TESTS_LOCATION="."
    SLURM_SCRIPT="slurm.jsc.sh"
    PARTITION="booster"

elif [[ $WHERE_TO_RUN == "vega" ]]; then

    # Vega (container)
    export MNIST_PATH="/ceph/hpc/data/st2301-itwin-users/mbunino/mnist"
    export CONTAINER_PATH="../../itwinai.sif"
    # Path to shared filesystem that all the Ray workers can access. /tmp is a local filesystem path to each worker
    # This is only needed by tests. This is needed by the torch trainer to store and retrieve checkpoints
    export SHARED_FS_PATH="/ceph/hpc/data/st2301-itwin-users/tmp-mbunino2"

    TESTS_LOCATION="/app/tests"
    SLURM_SCRIPT="slurm.vega.sh"
    PARTITION="gpu"

else
    echo "On what system are you running?"
    exit 1
fi

# Cleanup SLURM logs (*.out and *.err files) and other logs
rm -rf logs_slurm
mkdir logs_slurm
rm -rf logs_torchrun logs_mpirun logs_srun checkpoints


# # Launch tests
# export DIST_MODE="ray"
# export RUN_NAME="ray-itwinai"
# export COMMAND="pytest -v -s -o log_cli=true -o log_cli_level=INFO -m ray_dist $TESTS_LOCATION"
# sbatch  \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     --nodes=$NUM_NODES \
#     --gpus-per-node=$GPUS_PER_NODE \
#     --gres=gpu:$GPUS_PER_NODE \
#     --partition=$PARTITION \
#     $SLURM_SCRIPT

export DIST_MODE="ray"
export RUN_NAME="ray-itwinai"
export COMMAND='pytest -v -m ray_dist /app/tests'
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.vega.sh


export DIST_MODE="ddp"
export RUN_NAME="ddp-itwinai"
export COMMAND="pytest -v -s -o log_cli=true -o log_cli_level=INFO -m torch_dist $TESTS_LOCATION"
# export COMMAND="pytest -v -s -o log_cli=true -o log_cli_level=INFO -m torch_dist test_torch_trainer.py"
sbatch  \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    --nodes=$NUM_NODES \
    --gpus-per-node=$GPUS_PER_NODE \
    --gres=gpu:$GPUS_PER_NODE \
    --partition=$PARTITION \
    $SLURM_SCRIPT

# export DIST_MODE="deepspeed"
# export RUN_NAME="ds-itwinai"
# export COMMAND="pytest -v -s -o log_cli=true -o log_cli_level=INFO -m deepspeed_dist $TESTS_LOCATION"
# sbatch  \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     --nodes=$NUM_NODES \
#     --gpus-per-node=$GPUS_PER_NODE \
#     --gres=gpu:$GPUS_PER_NODE \
#     --partition=$PARTITION \
#     $SLURM_SCRIPT

# export DIST_MODE="horovod"
# export RUN_NAME="horovod-itwinai"
# export COMMAND="pytest -v -s -o log_cli=true -o log_cli_level=INFO -m horovod_dist $TESTS_LOCATION"
# sbatch \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     --nodes=$NUM_NODES \
#     --gpus-per-node=$GPUS_PER_NODE \
#     --gres=gpu:$GPUS_PER_NODE \
#     --partition=$PARTITION \
#     $SLURM_SCRIPT
