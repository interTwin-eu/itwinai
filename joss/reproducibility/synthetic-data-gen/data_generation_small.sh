#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
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

mkdir -p data

# Use the itwinai.sif container as the environment. Run only the first step in the pipeline
srun singularity exec itwinai.sif -B data:/data\
  itwinai exec-pipeline +pipe_key=training_pipeline_small +pipe_steps=[0]