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
#SBATCH --ntasks-per-node=1
# SBATCH --mem-per-gpu=10G
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ml --force purge
# ml Python CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/11.7
# ml GCCcore/11.3.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 cuDNN

ml Python
module unload OpenSSL

source ~/.bashrc

# Activate the environment
source ../../.venv-pytorch/bin/activate

GAN_DATASET="exp_data" #"/ceph/hpc/data/st2301-itwin-users/egarciagarcia"

# launch training
TRAINING_CMD="$(which itwinai) exec-pipeline num_nodes=$SLURM_NNODES \
    dataset_location=$GAN_DATASET "

srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
    $TRAINING_CMD "