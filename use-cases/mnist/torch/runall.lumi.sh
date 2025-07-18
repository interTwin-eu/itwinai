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
export CONTAINER_PATH="/scratch/project_465001592/itwinai-containers/itwinai-lumi-dev-2.sif"

# This is the location to local itwinai library files (outside the container), to be mounted inside the container
export ITWINAI_LOCATION_HOST="/project/project_465001592/buninoma/itwinai/src/" #"/users/$USER/itwinai/src/"

# Clear SLURM logs (*.out and *.err files)
read -p "Delete all existing scalability metrics and logs y/n?: " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf scalability-metrics logs_* checkpoints_* plots mllogs outputs ray_checkpoints
fi
mkdir -p logs_slurm logs_torchrun

export HYDRA_FULL_ERROR=1

# SLURM config
NNODES=2
NGPUS_PER_NODE=8
TOT_GPUS=$(( NNODES * NGPUS_PER_NODE ))

# DDP itwinai
DIST_MODE="ddp"
run_name="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_name=$run_name +pipe_key=training_pipeline ray_num_workers=$TOT_GPUS"
# TRAINING_CMD="itwinai check-distributed-cluster --platform amd --launcher ray"
sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_name-n$TOT_GPUS" \
    --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
    --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
    --nodes=$NNODES \
    --gpus-per-node=$NGPUS_PER_NODE \
    slurm.lumi.sh

# DeepSpeed itwinai
DIST_MODE="deepspeed"
run_name="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_name=$run_name +pipe_key=training_pipeline ray_num_workers=$TOT_GPUS"
sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_name-n$TOT_GPUS" \
    --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
    --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
    --nodes=$NNODES \
    --gpus-per-node=$NGPUS_PER_NODE \
    slurm.lumi.sh

# Horovod itwinai
DIST_MODE="horovod"
run_name="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd run_name=$run_name +pipe_key=training_pipeline ray_num_workers=$TOT_GPUS"
sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_name-n$TOT_GPUS" \
    --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
    --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
    --nodes=$NNODES \
    --gpus-per-node=$NGPUS_PER_NODE \
    slurm.lumi.sh

# ### GAN training ###

# # DDP itwinai
# DIST_MODE="ddp"
# run_name="$DIST_MODE-gan-${1:-itwinai}"
# TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_name=$run_name +pipe_key=training_pipeline_gan"
# sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_name-n$TOT_GPUS" \
#     --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
#     --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
#     --nodes=$NNODES \
#     --gpus-per-node=$NGPUS_PER_NODE \
#     slurm.lumi.sh

# # DeepSpeed itwinai
# DIST_MODE="deepspeed"
# run_name="$DIST_MODE-gan-${1:-itwinai}"
# TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_name=$run_name +pipe_key=training_pipeline_gan"
# sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_name="$run_name",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_name-n$TOT_GPUS" \
#     --output="logs_slurm/job-$run_name-n$TOT_GPUS.out" \
#     --error="logs_slurm/job-$run_name-n$TOT_GPUS.err" \
#     --nodes=$NNODES \
#     --gpus-per-node=$NGPUS_PER_NODE \
#     slurm.lumi.sh

# NOTE: GAN with Horovod does not work
