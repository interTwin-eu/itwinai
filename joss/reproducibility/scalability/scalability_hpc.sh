#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Clear SLURM logs (*.out and *.err files)
read -p "Delete all existing scalability metrics and logs y/n?: " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf scalability-metrics logs_* checkpoints_* plots mllogs outputs ray_checkpoints
fi
mkdir -p logs_slurm

export HYDRA_FULL_ERROR=1
DIST_BACKEND="nccl"
CONTAINER_PATH="itwinai.sif" # The Singularity container

# Launch multiple jobs for a given distributed strategy
run_scale(){
    if [ -z $1 ]; then
        echo "Please provide a distributed strategy... Unk"
        exit 1
    fi
    DIST_MODE="$1"
    for nnodes in 1 2 4; do
        # Prepare SLURM job for this number of nodes

        # SLURM config
        NGPUS_PER_NODE=4
        TOT_GPUS=$(( nnodes * NGPUS_PER_NODE ))
    
        TRAINING_CMD="itwinai exec-pipeline \
            strategy=$DIST_MODE \
            +pipe_key=training_pipeline_small \
            dist_backend=$DIST_BACKEND"
        
        # Submit job
        sbatch --export=ALL,DIST_MODE="$DIST_MODE",TRAINING_CMD="$TRAINING_CMD",CONTAINER_PATH="$CONTAINER_PATH" \
            --job-name="$DIST_MODE-n$TOT_GPUS" \
            --output="logs_slurm/job-$DIST_MODE-n$TOT_GPUS.out" \
            --error="logs_slurm/job-$DIST_MODE-n$TOT_GPUS.err" \
            --nodes=$nnodes \
            --gpus-per-node=$NGPUS_PER_NODE \
            scalability/slurm.jsc.sh
    done
}


# Run scaling test for torch DDP
run_scale "ddp"

# Run scaling test for deepspeed
run_scale "deepspeed"

# Run scaling test for horovod
run_scale "horovod"