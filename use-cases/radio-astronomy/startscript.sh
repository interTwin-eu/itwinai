#!/bin/bash

# general configuration of the job
#SBATCH --job-name=PrototypeTest
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:1

# load modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

# shellcheck source=/dev/null
source ~/.bashrc

# Using legacy (2.16) version of Keras
# Latest version with TF (2.16) installs Keras 3.3
# which returns an error for multi-node execution

# ON LOGIN NODE download datasets:
source /p/project1/intertwin/krochak1/itwinai/.venv/bin/activate
srun uv run dataloader.py
