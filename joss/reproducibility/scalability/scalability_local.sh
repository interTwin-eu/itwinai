#!/bin/bash

# Run this script on a single-host environment.

# Set to "nccl" if your machine has multiple GPUs.
DIST_BACKEND="gloo"

torchrun_launcher(){
    if [ -z $1 ]; then
        echo "How many workers for torchrun? Unk"
        exit 1
    fi
    if [ -z $2 ]; then
        echo "What distributed strategy with torchrun? Unk"
        exit 1
    fi
    docker run --rm -v "$PWD/data":/data -v "$PWD":/experiments --user "$UID" \
    ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    torchrun --no-python --standalone --nnodes=1 --nproc-per-node=$1 \
    itwinai exec-pipeline +pipe_key=training_pipeline_small \
        strategy=$2 dist_backend=$DIST_BACKEND
}

horovodrun_launcher(){
    if [ -z $1 ]; then
        echo "How many workers for torchrun? Unk"
        exit 1
    fi
    if [ -z $2 ]; then
        echo "What distributed strategy with torchrun? Unk"
        exit 1
    fi
    docker run --rm -v "$PWD/data":/data -v "$PWD":/experiments --user "$UID" \
    ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    horovodrun -np=$1 --mpi \
    itwinai exec-pipeline +pipe_key=training_pipeline_small \
        strategy=$2 dist_backend=$DIST_BACKEND
}

# Cleanup existing experiment results
rm -rf mllogs outputs checkpoints plots

# DDP scaling test
for num_workers in 2 3 4; do
    torchrun_launcher $num_workers "ddp"
    echo -e "\n#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*\n"
done

# # DeepSpeed scaling test
# for num_workers in 2 3 4; do
#     torchrun_launcher $num_workers "deepspeed"
#     echo -e "\n#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*\n"
# done

# Horovod scaling test
for num_workers in 2 3 4; do
    horovodrun_launcher $num_workers "horovod"
    echo -e "\n#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*\n"
done