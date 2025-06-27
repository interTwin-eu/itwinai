#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

export CONTAINER_PATH="/project/project_465001592/itwinai-containers/itwinai-dev.sif"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm checkpoints* mllogs* ray_checkpoints logs_torchrun
mkdir -p logs_slurm logs_torchrun

export HYDRA_FULL_ERROR=1

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.lumi.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai"
TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.lumi.sh

# Horovod itwinai
DIST_MODE="horovod"
RUN_NAME="horovod-itwinai"
TRAINING_CMD="itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.lumi.sh

### GAN training ###

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp +pipe_key=training_pipeline_gan"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.lumi.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai"
TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed +pipe_key=training_pipeline_gan"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    slurm.lumi.sh

# GAN with Horovod does not work
# Horovod itwinai
# DIST_MODE="horovod"
# RUN_NAME="horovod-itwinai"
# TRAINING_CMD="itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd +pipe_key=training_pipeline_gan"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.lumi.sh
