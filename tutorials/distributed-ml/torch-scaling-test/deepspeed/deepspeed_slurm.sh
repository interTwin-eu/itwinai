#!/bin/bash

# general configuration of the job
#SBATCH --job-name=Torch_DeepSpeed_tutorial-1
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job-ds.out
#SBATCH --error=job-ds.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# set modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# set env
source ../../../../envAI_hdfml/bin/activate

# job info
debug=false
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
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

# set env vars
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" -gt 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# launch training
MASTER_ADDR=$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)i
export MASTER_ADDR
export MASTER_PORT=29500 

TRAINING_CMD="DS_trainer.py -c config.yaml"

# Run without launcher: set --ntasks-per-node=NUM_GPUS
srun --cpu-bind=none python -u $TRAINING_CMD --deepspeed

# # Run with deepspeed launcher: set --ntasks-per-node=1
# # https://www.deepspeed.ai/getting-started/#multi-node-environment-variables
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# nodelist=$(scontrol show hostname $SLURM_NODELIST)
# echo "$nodelist" | sed -e 's/$/ slots=4/' > .hostfile
# # Requires passwordless SSH access among compute node
# srun --cpu-bind=none deepspeed --hostfile=.hostfile $TRAINING_CMD --deepspeed
# rm .hostfile

