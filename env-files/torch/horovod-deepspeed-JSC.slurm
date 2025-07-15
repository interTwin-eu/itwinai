#!/bin/bash

# Installation for JSC

# Job configuration
#SBATCH --job-name=setup_venv
#SBATCH --account=intertwin
#SBATCH --output=horovod_ds_installation.out
#SBATCH --error=horovod_ds_installation.err
#SBATCH --time=00:30:00

# Resource allocation
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu

ml --force purge
ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git

source .venv/bin/activate
source env-files/torch/install-horovod-deepspeed-cuda.sh
