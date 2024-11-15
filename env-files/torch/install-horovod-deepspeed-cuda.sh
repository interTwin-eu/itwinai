#!/bin/bash

# Job configuration
#SBATCH --job-name=setup_venv
#SBATCH --account=intertwin
#SBATCH --output=horovod_ds_installation.out
#SBATCH --error=horovod_ds_installation.err
#SBATCH --time=00:30:00

# Resources allocation
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu

ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git Clang

source .venv/bin/activate

# Horovod variables
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

uv pip install --no-cache-dir horovod[pytorch]

# DeepSpeed variables
export DS_BUILD_CCL_COMM=1
export DS_BUILD_UTILS=1
export DS_BUILD_AIO=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_FUSED_LAMB=1
export DS_BUILD_TRANSFORMER=1
export DS_BUILD_STOCHASTIC_TRANSFORMER=1
export DS_BUILD_TRANSFORMER_INFERENCE=1

uv pip install --no-cache-dir --no-build-isolation deepspeed
