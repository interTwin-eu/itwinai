#!/bin/bash

# Try MPI helloworld

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00

# Resources allocation
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# Load environment modules
# ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio  mpi4py
ml --force purge
export MODULEPATH=$HOME/.local/easybuild/modules/all:$MODULEPATH
ml Stages/2024 OpenMPI/4.1.5-GCC-12.3.0

# OUTSIDE: mpicc -o hw_out tmp_hello.c
# INSIDE: singularity exec itwinai_torch.sif mpicc -o hw_in tmp_hello.c

####################################################
### Using version complied outside the container ###
####################################################

echo "OUTSIDE CONTAINER"

srun --ntasks-per-node=4 ./hw_out

echo "OUT------=======------------------++++++----------------"
>&2 echo "OUT------=======------------------++++++----------------" 

srun --ntasks-per-node=4 --mpi=pmi2 singularity exec itwinai_torch.sif ./hw_out

echo "OUT------=======------------------++++++----------------"
>&2 echo "OUT------=======------------------++++++----------------" 

mpirun -np 4 singularity exec itwinai_torch.sif ./hw_out

echo "OUT------=======------------------++++++----------------"
>&2 echo "OUT------=======------------------++++++----------------" 

mpirun -np 4 ./hw_out

echo "OUT------=======------------------++++++----------------"
>&2 echo "OUT------=======------------------++++++----------------" 

####################################################
### Using version complied inside the container  ###
####################################################

echo "INSIDE CONTAINER"

srun --ntasks-per-node=4 ./hw_in

echo "IN------=======------------------++++++----------------"
>&2 echo "IN------=======------------------++++++----------------" 

srun --ntasks-per-node=4 --mpi=pmi2 singularity exec itwinai_torch.sif ./hw_in

echo "IN------=======------------------++++++----------------"
>&2 echo "IN------=======------------------++++++----------------" 

mpirun -np 4 singularity exec itwinai_torch.sif ./hw_in

echo "IN------=======------------------++++++----------------"
>&2 echo "IN------=======------------------++++++----------------" 

mpirun -np 4 ./hw_in

echo "IN------=======------------------++++++----------------"
>&2 echo "IN------=======------------------++++++----------------" 

####################################################
###                Lessons learned               ###
####################################################

# 1. mpirun -np 4 ./hw_out --> works
# 2. mpirun -np 4 singularity exec itwinai_torch.sif ./hw_in --> works when 
#    using my patched container
# 3. srun never works
# IMPORTANT: Lesson 2. is promising! Horovod could work in my patched container
#    if I had access to mpirun command! (..or maybe srun with PMIx support).
#    My patched container is itwinai_torch.sif

####################################################
###                      Other                   ###
####################################################

# # Works when using mpicc from lmod modules
# srun --ntasks-per-node=4 ./hw

# # Fails saying that MPI is not compatible with slurm
# srun --ntasks-per-node=4 --mpi=pmi2 singularity exec itwinai_torch.sif ./hw

# # # Fails
# mpirun -np 4 singularity exec itwinai_torch.sif ./hw

# When running binary which was compiled inside the container: Fails because /hw: /usr/lib64/libc.so.6: version `GLIBC_2.34' not found (required by ./hw)
# Success when compiling it outside
# mpirun -np 4 ./hw

