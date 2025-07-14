#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

set -e 

# DeepSpeed variables
export DS_BUILD_CCL_COMM=1
export DS_BUILD_UTILS=1
export DS_BUILD_AIO=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_FUSED_LAMB=1
export DS_BUILD_TRANSFORMER=1
export DS_BUILD_STOCHASTIC_TRANSFORMER=1
export DS_BUILD_TRANSFORMER_INFERENCE=1

# Use --no-cache-dir to avoid caching packages in your $HOME, which may have small disk quota
uv pip install --no-cache-dir --no-build-isolation "deepspeed==0.16.8"

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

uv pip install --no-cache-dir --no-build-isolation git+https://github.com/horovod/horovod.git@3a31d93 

echo "Finished Horovod and DeepSpeed installation script!"
