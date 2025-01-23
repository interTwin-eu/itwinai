#!/bin/bash

# SLURM jobscript for JSC systems

# general configuration of the job
#SBATCH --job-name=PrototypeTest
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# load modules
ml --force purge
ml Stages/2024 GCC CUDA/12 cuDNN Python 
# ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
# ml Python CMake HDF5 PnetCDF libaio mpi4py

# shellcheck source=/dev/null
source ~/.bashrc

# Activate the environment
source ../../envAI_hdfml/bin/activate

GAN_DATASET="exp_data" #"/p/scratch/intertwin/datasets/cern/"

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
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $TRAINING_CMD "