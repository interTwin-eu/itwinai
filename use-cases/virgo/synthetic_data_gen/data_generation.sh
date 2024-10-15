#!/bin/bash

#SBATCH --account=intertwin
#SBATCH --output=array_job.out
#SBATCH --error=array_job.err
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=batch
#SBATCH --array=1-250
#SBATCH --job-name=generate_virgo_data

# Load required modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Activate Python virtual environment
source ../../../envAI_hdfml/bin/activate

# Folder in which the datasets will be stored
target_folder="/p/scratch/intertwin/datasets/virgo/folder_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$target_folder"

# Set the number of files to generate. If NUM_FILES is not set, use 100 as the default value.
NUM_FILES=${NUM_FILES:-100}

# Run the Python script with appropriate arguments
srun python file_gen.py --target_folder_name "$target_folder" --file_number "$NUM_FILES"
