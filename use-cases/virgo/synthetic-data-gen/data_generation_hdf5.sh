#!/bin/bash

#SBATCH --account=intertwin
#SBATCH --output=array_job.out
#SBATCH --error=array_job.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=develbooster
#SBATCH --array=1-250
#SBATCH --job-name=generate_virgo_data

# Load required modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Activate Python virtual environment
source ../../envAI_hdfml/bin/activate

# Folder in which the datasets will be stored
target_file="/p/scratch/intertwin/datasets/virgo_hdf5/virgo_data_${SLURM_ARRAY_TASK_ID}.hdf5"

srun python synthetic_data_gen/file_gen_hdf5.py \
  --num-datapoints 3000 \
  --save-frequency 20 \
  --save-location "$target_file"
