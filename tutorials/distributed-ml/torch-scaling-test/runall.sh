#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Run all versions of distributed ML version

# This syntax means: "1" for first argument with "2" as a default value etc.
NUM_NODES=${1:-2}
TIMEOUT=${2:-"00:30:00"}

ACCOUNT="intertwin"
PARTITION="develbooster"
PYTHON_VENV="../../../.venv"
LOG_DIR="logs_slurm"

# Common options
CMD="--nodes=$N --time=$T --account=intertwin --partition=batch slurm.sh"
PYTHON_VENV="../../../envAI_hdfml"

echo "Distributing training over $N nodes. Timeout set to: $T"

mkdir -p $LOG_DIR
rm "$LOG_DIR"/*

rm *.csv # *checkpoint.pth.tar 

submit_job() {
    DIST_MODE=$1
    RUN_NAME=$2
    TRAINING_CMD=$3

    echo "Submitting job: $RUN_NAME with distribution mode: $DIST_MODE"
    sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
        --job-name="$RUN_NAME-n$NUM_NODES" \
        --output="$LOG_DIR/job-$RUN_NAME-n$NUM_NODES.out" \
        --error="$LOG_DIR/job-$RUN_NAME-n$NUM_NODES.err" \
        --nodes="$NUM_NODES" \
        --time="$TIMEOUT" \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        slurm.sh
}

# Submit jobs explicitly
submit_job "ddp" "ddp-bl-imagenet" "ddp_trainer.py -c config/base.yaml -c config/ddp.yaml"
# submit_job "deepspeed" "deepspeed-bl-imagenet" "deepspeed_trainer.py -c config/base.yaml -c config/deepspeed.yaml"
# submit_job "horovod" "horovod-bl-imagenet" "horovod_trainer.py -c config/base.yaml -c config/horovod.yaml"
# submit_job "ddp" "ddp-itwinai-imagenet" "itwinai_trainer.py -c config/base.yaml -c config/ddp.yaml -s ddp"
# submit_job "deepspeed" "deepspeed-itwinai-imagenet" "itwinai_trainer.py -c config/base.yaml -c config/deepspeed.yaml -s deepspeed"
# submit_job "horovod" "horovod-itwinai-imagenet" "itwinai_trainer.py -c config/base.yaml -c config/horovod.yaml -s horovod"
