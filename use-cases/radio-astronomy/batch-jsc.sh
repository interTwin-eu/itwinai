#!/bin/bash

## general configuration of the job
#SBATCH --job-name=itwinai-radio-astronomy
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=report.out
#SBATCH --error=progress.out
#SBATCH --time=02:00:00

## configure node and process count on the CM
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive

##########################################################
###  ^^^ You will need to change the values above ^^^  ###
### This batch script was tested on JSC Juwels-Booster ###
##########################################################

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT=54123

## load modules: configuration for JSC Juwels-Booster
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

## source the itwinai Python virtual environment 
source ../../.venv/bin/activate  ## <-- YOU MIGHT NEED TO CHANGE THIS PATH

if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi

srun_torchrun() {
  srun --cpu-bind=none --ntasks-per-node=1 bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_endpoint=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n1):${MASTER_PORT} \
    $*"
}

## Data generation - best to run on a single node without GPUs
srun $(which itwinai) exec-pipeline +pipe_key=syndata_pipeline 

## Newtork training and evaluation - can be executed on multi-node / multi-GPU
srun_torchrun $(which itwinai) exec-pipeline +pipe_key=unet_pipeline
srun_torchrun $(which itwinai) exec-pipeline +pipe_key=fcnn_pipeline
srun_torchrun $(which itwinai) exec-pipeline +pipe_key=cnn1d_pipeline
srun_torchrun $(which itwinai) exec-pipeline +pipe_key=evaluate_pipeline
