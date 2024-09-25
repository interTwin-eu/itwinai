#!/bin/bash

NUM_NODES=2
NUM_GPUS=2
EPOCHS=2
TIME=0:10:00

sbatch --export=ALL,STRATEGY="ddp",EPOCHS="$EPOCHS" \
	--output="logs_slurm/job-ddp-n$NUM_NODES.out" \
	--error="logs_slurm/job-ddp-n$NUM_NODES.err" \
	--nodes=$NUM_NODES \
	--time=$TIME \
	ddp_deepspeed_slurm.sh

sbatch --export=ALL,STRATEGY="deepspeed",EPOCHS="$EPOCHS" \
	--output="logs_slurm/job-deepspeed-n$NUM_NODES.out" \
	--error="logs_slurm/job-deepspeed-n$NUM_NODES.err" \
	--nodes=$NUM_NODES \
	--time=$TIME \
	ddp_deepspeed_slurm.sh

sbatch --export=ALL,EPOCHS="$EPOCHS" \
	--output="logs_slurm/job-horovod-n$NUM_NODES.out" \
	--error="logs_slurm/job-horovod-n$NUM_NODES.err" \
	--nodes=$NUM_NODES \
	--time=$TIME \
	horovod_slurm.sh

# Python virtual environment (no conda/micromamba)
#PYTHON_VENV="../../envAI_hdfml"
# CMD="--nodes=$NUM_NODES --time=$TIME --account=intertwin --partition=batch slurm.sh"

# echo "Distributing training over $N nodes. Timeout set to: $T"

# Clear SLURM logs (*.out and *.err files)
# rm -rf logs_slurm checkpoints*
# mkdir logs_slurm
# rm -rf logs_torchrun
# # Clear scaling test logs 
# rm *.csv

# DDP itwinai
# DIST_MODE="ddp"
# RUN_NAME="ddp-itwinai"
# TRAINING_CMD="torch_dist_final_scaling.py --strategy ddp"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     $CMD
#
# # DeepSpeed itwinai
# DIST_MODE="deepspeed"
# RUN_NAME="deepspeed-itwinai"
# TRAINING_CMD="python torch_dist_final_scaling.py --strategy deepspeed"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     $CMD

# Horovod itwinai
# DIST_MODE="horovod"
# RUN_NAME="horovod-itwinai"
# TRAINING_CMD="python torch_dist_final_scaling.py --strategy horovod"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     $CMD
