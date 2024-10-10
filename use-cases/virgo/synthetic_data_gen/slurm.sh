#!/bin/bash

#SBATCH --account=intertwin
#SBATCH --output=array_job.out
#SBATCH --error=array_job.err
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=batch
#SBATCH --array=1-250
#SBATCH --job-name=generate_virgo_data

# Activate Python virtual env
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py
source ../../../envAI_hdfml/bin/activate

mkdir /p/scratch/intertwin/datasets/virgo/folder_${SLURM_ARRAY_TASK_ID}
folder_name="/p/scratch/intertwin/datasets/virgo/folder_${SLURM_ARRAY_TASK_ID}"
srun python /p/project1/intertwin/mutegeki1/itwinai/use-cases/virgo/synthetic_data_gen/file_gen.py "$folder_name" "${1}"