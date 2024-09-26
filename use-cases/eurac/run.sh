#!/bin/bash

# Python virtual environment (no conda/micromamba)
# PYTHON_VENV="../../../hython-dev"
PYTHON_VENV="../../envAI_hdfml"

if [ -z "$1" ]; then
    CONFIG_FILE=config.yaml
		>&2 echo "WARNING (run.sh): env variable CONFIG_FILE is not set. Defaulting to $CONFIG_FILE."
else
    CONFIG_FILE=$1
fi
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config "$CONFIG_FILE" --pipe-key training_pipeline" #-o strategy=ddp -o checkpoint_path=checkpoints_ddp/epoch_{}.pth
NUM_NODES=1

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm checkpoints*
mkdir logs_slurm
rm -rf logs_torchrun

# DDP itwinai
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV",CONFIG_FILE=$CONFIG_FILE \
    --job-name="$RUN_NAME-n$NUM_NODES" \
      --nodes="$NUM_NODES" \
    --output="logs_slurm/job-$RUN_NAME-n$NUM_NODES.out" \
    --error="logs_slurm/job-$RUN_NAME-n$NUM_NODES.err" \
    slurm.sh

# # DeepSpeed itwinai
# DIST_MODE="deepspeed"
# RUN_NAME="deepspeed-itwinai"
# TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1: -o strategy=deepspeed -o checkpoint_path=checkpoints_deepspeed/epoch_{}.pth"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.sh

# # Horovod itwinai
# DIST_MODE="horovod"
# RUN_NAME="horovod-itwinai"
# TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1: -o strategy=horovod -o checkpoint_path=checkpoints_horovod/epoch_{}.pth"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     slurm.sh
