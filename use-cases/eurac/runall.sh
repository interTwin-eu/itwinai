#!/bin/bash
# Run all versions of distributed ML version
# $1 (Optional[int]): number of nodes. Default: 2
# $2 (Optional[str]): timeout. Default: "00:30:00"

if [ -z "$1" ] ; then
    N=2
else
    N=$1
fi
if [ -z "$2" ] ; then
    T="00:30:00"
else
    T=$2
fi
# Python virtual environment (no conda/micromamba)
#PYTHON_VENV="../../envAI_hdfml"
PYTHON_VENV="../../../hython-dev"
CMD="--nodes=$N --time=$T --account=intertwin --partition=batch slurm.sh"

echo "Distributing training over $N nodes. Timeout set to: $T"

# Clear SLURM logs (*.out and *.err files)
rm -rf logs_slurm checkpoints*
mkdir logs_slurm
rm -rf logs_torchrun
# Clear scaling test logs 
rm *.csv

# DDP itwinai
DIST_MODE="ddp"
RUN_NAME="ddp-itwinai"
TRAINING_CMD="torch_dist_final_scaling.py --strategy ddp"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    $CMD

# DeepSpeed itwinai
DIST_MODE="deepspeed"
RUN_NAME="deepspeed-itwinai"
TRAINING_CMD="python torch_dist_final_scaling.py --strategy deepspeed"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    --job-name="$RUN_NAME-n$N" \
    --output="logs_slurm/job-$RUN_NAME-n$N.out" \
    --error="logs_slurm/job-$RUN_NAME-n$N.err" \
    $CMD

# Horovod itwinai
# DIST_MODE="horovod"
# RUN_NAME="horovod-itwinai"
# TRAINING_CMD="python torch_dist_final_scaling.py --strategy horovod"
# sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
#     --job-name="$RUN_NAME-n$N" \
#     --output="logs_slurm/job-$RUN_NAME-n$N.out" \
#     --error="logs_slurm/job-$RUN_NAME-n$N.err" \
#     $CMD