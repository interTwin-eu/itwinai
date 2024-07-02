#!/bin/bash

# Build the documentation locally and serve it on localhost on JSC systems

ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py
ml Stages/2023 Pandoc/2.19.2

source .venv-docs/bin/activate
cd docs
make clean && make html && python -m http.server -d _build/html