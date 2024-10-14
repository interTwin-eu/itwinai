#!/bin/bash

# SLURM jobscript for JSC systems

# Job configuration
#SBATCH --job-name=comm_test
#SBATCH --account=intertwin
#SBATCH --output=logs_slurm/deepspeed.out
#SBATCH --error=logs_slurm/deepspeed.err
#SBATCH --time=00:30:00

# Resources allocation
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --exclusive

# Load environment modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

PYTHON_VENV="../envAI_hdfml"
source $PYTHON_VENV/bin/activate

EPOCHS=2
TORCHRUN_STRATEGY="deepspeed"

run_torchrun() {
  srun --cpu-bind=none --ntasks-per-node=1 \
      bash -c "torchrun \
      --log_dir='logs_torchrun' \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$SLURM_GPUS_PER_NODE \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
      profiler.py --epochs $EPOCHS --strategy $TORCHRUN_STRATEGY"
}

run_horovod() {
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  srun --cpu-bind=none --nodes=$SLURM_NNODES \
        --ntasks-per-node=$SLURM_GPUS_PER_NODE \
        --cpus-per-task=$SLURM_CPUS_PER_GPU \
        python profiler.py --epochs $EPOCHS --strategy horovod
}

run_torchrun
# run_horovod

