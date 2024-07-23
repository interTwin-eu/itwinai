#!/bin/bash

# SLURM jobscript for Vega systems

# Job configuration
#SBATCH --job-name=3dgan_training
#SBATCH --account=s24r05-03-users
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=01:00:00

# Resources allocation
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --exclusive
#SBATCH --mem 40G

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

source ~/.bashrc

source ../../.venv-pytorch/bin/activate

# launch training
TRAINING_CMD="$(which itwinai) exec-pipeline --config config.yaml --pipe-key training_pipeline"

srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $TRAINING_CMD "