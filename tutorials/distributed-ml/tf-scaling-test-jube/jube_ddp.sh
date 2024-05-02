#!/bin/bash

# general configuration of the job
#SBATCH --job-name=JUBE_DDP
#SBATCH --account=#ACC#
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=#TIMELIM#

# configure node and process count on the CM
#SBATCH --partition=#QUEUE#
#SBATCH --nodes=#NODES#
#SBATCH --cpus-per-task=#NW#
#SBATCH --gpus-per-node=#NGPU#
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

set -x
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# set modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py CMake  cuDNN/8.9.5.29-CUDA-12

# set env
source /p/project/intertwin/rakesh/repo_push/itwinai/envAItf_hdfml/bin/activate

# Using legacy (2.16) version of Keras
# Latest version with TF (2.16) installs Keras 3.3
# which returns an error for multi-node execution
export TF_USE_LEGACY_KERAS=1

# sleep a sec
sleep 1

# job info
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "DEBUG: SLURM_NODELIST: $SLURM_NODELIST"
echo

# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" -gt 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

dataDir='/p/scratch/intertwin/datasets/imagenet/'

COMMAND="train.py"

EXEC="$COMMAND \
    --data_dir $dataDir"

srun python -u $EXEC


#eof
