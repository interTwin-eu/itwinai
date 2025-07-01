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
mkdir -p logs_slurm logs_torchrun

export HYDRA_FULL_ERROR=1

# DDP itwinai
DIST_MODE="ddp"
run_id="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_id=$run_id"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_id-n$N" \
    --output="logs_slurm/job-$run_id-n$N.out" \
    --error="logs_slurm/job-$run_id-n$N.err" \
    slurm.lumi.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
run_id="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_id=$run_id"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_id-n$N" \
    --output="logs_slurm/job-$run_id-n$N.out" \
    --error="logs_slurm/job-$run_id-n$N.err" \
    slurm.lumi.sh

# Horovod itwinai
DIST_MODE="horovod"
run_id="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd run_id=$run_id"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_id-n$N" \
    --output="logs_slurm/job-$run_id-n$N.out" \
    --error="logs_slurm/job-$run_id-n$N.err" \
    slurm.lumi.sh

### GAN training ###

# DDP itwinai
DIST_MODE="ddp"
run_id="$DIST_MODE-gan-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_id=$run_id +pipe_key=training_pipeline_gan"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_id-n$N" \
    --output="logs_slurm/job-$run_id-n$N.out" \
    --error="logs_slurm/job-$run_id-n$N.err" \
    slurm.lumi.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
run_id="$DIST_MODE-gan-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_id=$run_id +pipe_key=training_pipeline_gan"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_id-n$N" \
    --output="logs_slurm/job-$run_id-n$N.out" \
    --error="logs_slurm/job-$run_id-n$N.err" \
    slurm.lumi.sh

# GAN with Horovod does not work
# Horovod itwinai
# DIST_MODE="horovod"
# run_id="$DIST_MODE-gan-${1:-itwinai}"
# TRAINING_CMD="itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd run_id=$run_id +pipe_key=training_pipeline_gan"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_id-n$N" \
#     --output="logs_slurm/job-$run_id-n$N.out" \
#     --error="logs_slurm/job-$run_id-n$N.err" \
#     slurm.lumi.sh
