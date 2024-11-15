#!/bin/bash

#SBATCH --account=intertwin
#SBATCH --output=array-job/job_%a.out
#SBATCH --error=array-job/job_%a.err
#SBATCH --time=00:07:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=develbooster
#SBATCH --array=1-75
#SBATCH --job-name=generate_virgo_data
#SBATCH --cpus-per-task=26

# Load required modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Activate Python virtual environment
source ../../envAI_juwels/bin/activate

# Folder in which the datasets will be stored
target_file="/p/scratch/intertwin/datasets/virgo_hdf5/virgo_data_${SLURM_ARRAY_TASK_ID}.hdf5"

python synthetic-data-gen/file_gen_hdf5.py \
  --num-datapoints 10000 \
  --num-processes 25 \
  --save-frequency 1000 \
  --save-location "$target_file"

