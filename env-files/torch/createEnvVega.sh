#!/bin/bash
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

if [ ! -f "env-files/torch/generic_torch.sh" ]; then
  echo "ERROR: env-files/torch/generic_torch.sh not found!"
  exit 1
fi

# Load modules
# NOTE: REFLECT THEM IN THE MAIN README! 
ml --force purge
ml CMake/3.29.3-GCCcore-13.3.0
ml mpi4py/3.1.5
ml OpenMPI/4.1.6-GCC-13.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml CUDA/12.6.0
ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
ml Python/3.12.3-GCCcore-13.3.0

# You should have CUDA 12.6 now


# Create and install torch env
export ENV_NAME=".venv-pytorch"
export PIP_INDEX_TORCH_CUDA="https://download.pytorch.org/whl/cu126"
bash env-files/torch/generic_torch.sh
