#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Jarl Sondre Sæther
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

#SBATCH --account=intertwin
#SBATCH --output=array-job/job_%a.out
#SBATCH --error=array-job/job_%a.err
#SBATCH --time=00:07:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=develbooster
#SBATCH --array=1-75
#SBATCH --job-name=generate_virgo_data
#SBATCH --cpus-per-task=26

# Load software modules (JUWELS system)
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Location where the datasets will be stored
mkdir -p virgo_hdf5
target_file="virgo_hdf5/virgo_data_${SLURM_ARRAY_TASK_ID}.hdf5"

# Use the itwinai.sif container as the environment
srun singularity exec itwinai.sif \
  python synthetic-data-gen/file_gen_hdf5.py \
  --num-datapoints 10000 \
  --num-processes 25 \
  --save-frequency 1000 \
  --save-location "$target_file"

