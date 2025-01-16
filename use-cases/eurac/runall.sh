#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------
 
# Script for running all the distributed trainings with a certain number of 
# GPUs on a certain number of compute nodes. Can be run as a stand-alone script or 
# from another script. Change the values of the script either by changing the default 
# values below or by exporting environment variables before running the script. 

if [ -z "$NUM_NODES" ]; then 
	NUM_NODES=2
fi
if [ -z "$NUM_GPUS" ]; then 
	NUM_GPUS=4
fi
if [ -z "$TIME" ]; then 
	TIME=0:40:00
fi
if [ -z "$DEBUG" ]; then 
	DEBUG=false
fi
if [ -z "$PYTHON_VENV" ]; then 
	PYTHON_VENV="../../.venv/"
fi

submit_job () {
    local mode=$1
    sbatch --export=ALL,DIST_MODE="$mode",RUN_NAME="$mode",TIME="$TIME",DEBUG="$DEBUG",PYTHON_VENV=$PYTHON_VENV \
        --job-name="eurac-$mode" \
        --output="logs_slurm/job-$mode-n$NUM_NODES.out" \
        --error="logs_slurm/job-$mode-n$NUM_NODES.err" \
        --nodes="$NUM_NODES" \
        --gpus-per-node="$NUM_GPUS" \
        --time="$TIME" \
        slurm.sh
}

echo "Running distributed training on $NUM_NODES nodes with $NUM_GPUS GPUs per node"
submit_job "ddp"
submit_job "deepspeed"
submit_job "horovod"
