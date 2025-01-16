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
ml Python/3.11.5-GCCcore-13.2.0 
ml CMake/3.24.3-GCCcore-11.3.0
ml mpi4py
ml OpenMPI
ml CUDA/12.3
ml GCCcore/11.3.0
ml NCCL
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0

# You should have CUDA 12.3 now


# Create and install torch env
export ENV_NAME=".venv-pytorch"
export PIP_INDEX_TORCH_CUDA="https://download.pytorch.org/whl/cu121"
bash env-files/torch/generic_torch.sh
