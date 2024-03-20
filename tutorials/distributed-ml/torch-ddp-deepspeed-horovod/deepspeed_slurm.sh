#!/bin/bash

# general configuration of the job
#SBATCH --job-name=Torch_DeepSpeed_tutorial
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00

# configure node and process count on the CM
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
# SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# set modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# set env
source ../../../envAI_hdfml/bin/activate

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
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

# # set comm
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export OMP_NUM_THREADS=1
# if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
#   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# fi
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# launch training
TRAINING_CMD="train.py -s deepspeed"

# This command does not integrate well with torch.distributed
# because, e.g., global rank is not recognized -> all processes print to console.
# It raises the error: AssertionError: LOCAL_RANK (2) != OMPI_COMM_WORLD_LOCAL_RANK (0), not sure how to proceed as we're seeing conflicting local rank info.
srun --cpu-bind=none bash -c "deepspeed $TRAINING_CMD"


# srun python -m deepspeed.launcher.launch \
#      --node_rank $SLURM_PROCID \
#      --master_addr ${SLURMD_NODENAME}i \
#      --master_port 29500 \
#      --world_info $WID \
#      $TRAINING_CMD --deepspeed_mpi 


