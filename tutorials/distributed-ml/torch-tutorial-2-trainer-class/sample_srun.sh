#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------
# This file contains the sample bash code that was used in the interTwin presentation
# held on Feb. 18. It is meant to illustrate how to combine srun and torchrun to launch
# processes in parallel that can communicate and thus facilitate distributed ML. 

srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=151152 \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    python train.py"
