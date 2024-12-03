#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# Run tests on JSC environment
# Set TORCH_ENV and TF_ENV variables below to use different
# virtual environment names.

ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio

export TORCH_ENV="envAI_hdfml"
export TF_ENV="envAItf_hdfml"

if [ ! -d "$TORCH_ENV" ]; then
  echo "$TORCH_ENV not found!"
  exit 1
fi
if [ ! -d "$TF_ENV" ]; then
  echo "$TF_ENV not found!"
  exit 1
fi

# Avoid downloading datasets from Gdrive
export CERN_DATASET="/p/project1/intertwin/smalldata/3dgan-sample"
export CMCCC_DATASET="/p/project1/intertwin/smalldata/cmcc"
export MNIST_DATASET="/p/project1/intertwin/smalldata/mnist"

$TORCH_ENV/bin/pytest -v tests/ -m "not slurm"