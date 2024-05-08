#!/bin/bash

# general configuration of the job
#SBATCH --job-name=cyclones
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

# SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

set -x
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# load modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

source ../../envAItf_hdfml/bin/activate

# job info
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "DEBUG: SLURM_NODELIST: $SLURM_NODELIST"
echo

# ONLY IF TENSORFLOW >= 2.16:
# # Using legacy (2.16) version of Keras
# # Latest version with TF (2.16) installs Keras 3.3
# # which returns an error for multi-node execution
# export TF_USE_LEGACY_KERAS=1

# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" -gt 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# ON LOGIN NODE download datasets:
# ../../.venv-tf/bin/python train.py -p pipeline.yaml --download-only

# --data_path argument is optional, but on JSC we use the dataset we previously downloaded
srun python train.py -p pipeline.yaml --data_path /p/project/intertwin/smalldata/cmcc