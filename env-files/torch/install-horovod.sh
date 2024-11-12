#!/bin/bash

# Job configuration
#SBATCH --job-name=setup_venv
#SBATCH --account=intertwin
#SBATCH --output=logs_slurm/job.out
#SBATCH --error=logs_slurm/job.err
#SBATCH --time=00:10:00

# Resources allocation
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu

ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git

source .venv/bin/activate
export LDSHARED="$CC -shared" &&
export CMAKE_CXX_STANDARD=17 

export HOROVOD_MPI_THREADS_DISABLE=1
export HOROVOD_CPU_OPERATIONS=MPI

export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_NCCL_LINK=SHARED
export HOROVOD_NCCL_HOME=$EBROOTNCCL

export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1

uv pip install --no-cache-dir --no-build-isolation git+https://github.com/horovod/horovod.git
