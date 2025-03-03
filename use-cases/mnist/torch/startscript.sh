#!/bin/bash

# general configuration of the job
#SBATCH --job-name=PrototypeTest
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:05:00

# configure node and process count on the CM
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# load modules
ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python/3.11.3 HDF5 PnetCDF libaio mpi4py

# shellcheck source=/dev/null
# source ~/.bashrc

# ON LOGIN NODE download datasets:
# ../../../.venv-pytorch/bin/itwinai exec-pipeline +pipe_steps=[0]
source ../../../.venv/bin/activate
srun itwinai exec-pipeline
