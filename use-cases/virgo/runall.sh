#!/bin/bash

# Python virtual environment (no conda/micromamba)
PYTHON_VENV="../../envAI_hdfml"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm checkpoints*
mkdir logs_slurm
rm -rf logs_torchrun

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1: -o strategy=ddp -o checkpoint_path=checkpoints_ddp/epoch_{}.pth"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1: -o strategy=deepspeed -o checkpoint_path=checkpoints_deepspeed/epoch_{}.pth"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh

# Horovod itwinai
DIST_MODE="horovod"
RUN_NAME="horovod-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1: -o strategy=horovod -o checkpoint_path=checkpoints_horovod/epoch_{}.pth"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.sh