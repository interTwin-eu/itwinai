#!/bin/bash

# Create .venv-docs virtualenv to build the documentation locally on JSC systems

ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py

cmake --version
gcc --version

rm -rf .venv-docs
python -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r docs/pre-requirements.txt
pip install -r docs/requirements.txt