#!/bin/bash
# -*- coding: utf-8 -*-
# version: 2212008a
# pull and build containers for PyTorch/NVIDIA

# load modules
ml GCC/11.3.0 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Apptainer-Tools/2023

# create Cache/TMP so that $HOME would not be used
mkdir -p Cache
mkdir -p TMP 
export APPTAINER_CACHEDIR=$(mktemp -d -p $PWD/Cache)
export APPTAINER_TMPDIR=$(mktemp -d -p $PWD/TMP)

# official NVIDIA NVCR container with Torch==2.0.0
apptainer pull containers/apptainer/itwinai.sif docker://nvcr.io/nvidia/pytorch:23.09-py3

# run bash to create envs
echo "running ./containers/apptainer/apptainer_build_env.sh"
apptainer exec itwinai.sif bash -c "./containers/apptainer/apptainer_build_env.sh"

#eof