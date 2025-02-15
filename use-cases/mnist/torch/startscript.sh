#!/bin/bash

# general configuration of the job
#SBATCH --job-name=PrototypeTest
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# load modules
ml --force purge
ml Stages/2023 StdEnv/2023 NVHPC/23.1 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Python/3.10.4 HDF5 libaio/0.3.112 GCC/11.3.0

# shellcheck source=/dev/null
source ~/.bashrc

# ON LOGIN NODE download datasets:
# ../../../.venv-pytorch/bin/itwinai exec-pipeline +pipe_steps=[0]
source ../../../.venv-pytorch/bin/activate
srun itwinai exec-pipeline