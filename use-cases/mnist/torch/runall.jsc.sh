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

# Python virtual environment (no conda/micromamba)
PYTHON_VENV="../../../.venv"


# Clear SLURM logs (*.out and *.err files)
read -p "Delete all existing scalability metrics and logs y/n?: " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf scalability-metrics logs_* checkpoints_* plots mllogs outputs ray_checkpoints
fi
mkdir -p logs_slurm

export HYDRA_FULL_ERROR=1

# SLURM config
NNODES=1
NGPUS_PER_NODE=4
TOT_GPUS=$(( NNODES * NGPUS_PER_NODE ))

# DDP itwinai
DIST_MODE="ddp"
run_name="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_name=$run_name +pipe_key=training_pipeline ray_num_workers=$TOT_GPUS"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_name-n$TOT_GPUS" \
    --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
    --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
    --nodes=$NNODES \
    --gpus-per-node=$NGPUS_PER_NODE \
    slurm.jsc.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
run_name="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_name=$run_name +pipe_key=training_pipeline ray_num_workers=$TOT_GPUS"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_name-n$TOT_GPUS" \
    --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
    --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
    --nodes=$NNODES \
    --gpus-per-node=$NGPUS_PER_NODE \
    slurm.jsc.sh

# Horovod itwinai
DIST_MODE="horovod"
run_name="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd run_name=$run_name +pipe_key=training_pipeline ray_num_workers=$TOT_GPUS"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_name-n$TOT_GPUS" \
    --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
    --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
    --nodes=$NNODES \
    --gpus-per-node=$NGPUS_PER_NODE \
    slurm.jsc.sh

### GAN training ###

# # DDP itwinai
# DIST_MODE="ddp"
# run_name="$DIST_MODE-gan-${1:-itwinai}"
# TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_name=$run_name +pipe_key=training_pipeline_gan ray_num_workers=$TOT_GPUS"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_name-n$N" \
#     --output="logs_slurm/job-$run_name-n$N.out" \
#     --error="logs_slurm/job-$run_name-n$N.err" \
#     slurm.jsc.sh

# # DeepSpeed itwinai
# DIST_MODE="deepspeed"
# run_name="$DIST_MODE-gan-${1:-itwinai}"
# TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_name=$run_name +pipe_key=training_pipeline_gan ray_num_workers=$TOT_GPUS"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_name-n$N" \
#     --output="logs_slurm/job-$run_name-n$N.out" \
#     --error="logs_slurm/job-$run_name-n$N.err" \
#     slurm.jsc.sh

# NOTE: GAN with Horovod does not work!
