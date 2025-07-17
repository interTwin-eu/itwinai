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
export CONTAINER_PATH="/projects/EEHPC-DEV-2024D11-012/matbun/containers/0.3.3-skinny-torch2.6-bookworm-a20bd15128334eed1296ca8f409f2b9180fc9eaf.sif"

# This is the location to local itwinai library files (outside the container), to be mounted inside the container
export ITWINAI_DYNAMIC_BIND="/projects/EEHPC-DEV-2024D11-012/matbun/itwinai/src/itwinai:/app/.venv/lib/python3.12/site-packages/itwinai"

# Clear SLURM logs (*.out and *.err files)
read -p "Delete all existing scalability metrics and logs y/n?: " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf scalability-metrics logs_* checkpoints_* plots mllogs outputs ray_checkpoints
fi
mkdir -p logs_slurm logs_torchrun

export HYDRA_FULL_ERROR=1

# SLURM config
NNODES=2
export NWORKERS_PER_NODE=4
NCPUS_PER_WORKER=8
NCPUS_PER_NODE=$(( NWORKERS_PER_NODE * NCPUS_PER_WORKER + 1)) # add one for the tuner process
TOT_WORKERS=$(( NWORKERS_PER_NODE * NNODES ))

# DDP itwinai
DIST_MODE="ddp"
run_id="$DIST_MODE-${1:-itwinai}"
TRAINING_CMD="itwinai exec-pipeline strategy=ddp checkpoints_location=checkpoints_ddp run_id=$run_id +pipe_key=training_pipeline \
              communication_backend=gloo \
              ray_num_workers=$TOT_WORKERS \
              ray_use_gpu=false ray_gpus_per_worker=0 \
              ray_cpus_per_worker=$(( NCPUS_PER_WORKER - 1 ))" # remove one for the trial process
TRAINING_CMD="itwinai check-distributed-cluster --platform arm --launcher torchrun"
sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$run_id-n$TOT_WORKERS" \
    --output="logs_slurm/job-$run_id-n$TOT_WORKERS.out" \
    --error="logs_slurm/job-$run_id-n$TOT_WORKERS.err" \
    --cpus-per-task=$NCPUS_PER_NODE \
    --nodes=$NNODES \
    slurm.deucalion.sh

# # DeepSpeed itwinai
# DIST_MODE="deepspeed"
# run_id="$DIST_MODE-${1:-itwinai}"
# TRAINING_CMD="itwinai exec-pipeline strategy=deepspeed checkpoints_location=checkpoints_deepspeed run_id=$run_id +pipe_key=training_pipeline \
#               communication_backend=gloo \
#               ray_num_workers=$TOT_WORKERS \
#               ray_use_gpu=false ray_gpus_per_worker=0 \
#               ray_cpus_per_worker=$(( NCPUS_PER_WORKER - 1 ))" # remove one for the trial process
# sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_id-n$TOT_WORKERS" \
#     --output="logs_slurm/job-$run_id-n$TOT_WORKERS.out" \
#     --error="logs_slurm/job-$run_id-n$TOT_WORKERS.err" \
#     --cpus-per-task=$NCPUS_PER_NODE \
#     --nodes=$NNODES \
#     slurm.deucalion.sh

# # Horovod itwinai
# DIST_MODE="horovod"
# run_id="$DIST_MODE-${1:-itwinai}"
# TRAINING_CMD="itwinai exec-pipeline strategy=horovod checkpoints_location=checkpoints_hvd run_id=$run_id +pipe_key=training_pipeline \
#               communication_backend=gloo \
#               ray_num_workers=$TOT_WORKERS \
#               ray_use_gpu=false ray_gpus_per_worker=0 \
#               ray_cpus_per_worker=$(( NCPUS_PER_WORKER - 1 ))" # remove one for the trial process
# sbatch --export=ALL,CONTAINER_PATH="$CONTAINER_PATH",DIST_MODE="$DIST_MODE",run_id="$run_id",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$run_id-n$TOT_WORKERS" \
#     --output="logs_slurm/job-$run_id-n$TOT_WORKERS.out" \
#     --error="logs_slurm/job-$run_id-n$TOT_WORKERS.err" \
#     --cpus-per-task=$NCPUS_PER_NODE \
#     --nodes=$NNODES \
#     slurm.deucalion.sh

