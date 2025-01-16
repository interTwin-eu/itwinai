#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Run all versions of distributed ML version
# $1 (Optional[int]): number of nodes. Default: 2
# $2 (Optional[str]): timeout. Default: "00:30:00"

if [ -z "$1" ] ; then
    N=2
else
    N=$1
fi
if [ -z "$2" ] ; then
    T="00:20:00"
else
    T=$2
fi

# Python virtual environment (no conda/micromamba)
CMD="--nodes=$N --time=$T --account=intertwin --partition=develbooster slurm.sh"
PYTHON_VENV="../../.venv/"

echo "Distributing training over $N nodes. Timeout set to: $T"
# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm checkpoints*
mkdir logs_slurm
rm -rf logs_torchrun

# Clear scaling test logs 
rm *.csv

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline -o strategy=ddp -o checkpoint_path=checkpoints_ddp/epoch_{}.pth"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="virgo-$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    $CMD

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline -o strategy=deepspeed -o checkpoint_path=checkpoints_deepspeed/epoch_{}.pth"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="virgo-$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    $CMD

# Horovod itwinai
DIST_MODE="horovod"
RUN_NAME="horovod-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline -o strategy=horovod -o checkpoint_path=checkpoints_horovod/epoch_{}.pth"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="virgo-$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    $CMD
