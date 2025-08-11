#!/bin/bash

## general configuration of the job
#SBATCH --job-name=itwinai-radio-astronomy
#SBATCH --account=slfse
#SBATCH --partition=devel
#SBATCH --output=/p/project1/intertwin/krochak1/pulsar-plugin/pytest-runner/actions-runner-logs/output.%j.out
#SBATCH --error=/p/project1/intertwin/krochak1/pulsar-plugin/pytest-runner/actions-runner-logs/error.%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py
ml X11/20230603 OpenGL/2023a

#shellcheck disable=SC1091
source /p/project1/intertwin/krochak1/pulsar-plugin/.venv-juwels-cluster/bin/activate

cd /p/project1/intertwin/krochak1/pulsar-plugin || exit
pytest tests