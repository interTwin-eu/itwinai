#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=lQCD
#SBATCH --account=intertwin
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:10:00

# Node and process configuration
#SBATCH --partition=develbooster
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --exclusive

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# Load necessary modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Activate the virtual environment
source ../../.venv/bin/activate

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    export MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

# Set CUDA devices and OpenMP threads
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" -gt 0 ] ; then
   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# Read strategy from config.yaml
STRATEGY=$(grep '^strategy:' config.yaml | sed -E "s/.*strategy:[[:space:]]*'?([^']*)'?/\1/")

echo "Launching training with strategy: $STRATEGY"

case "$STRATEGY" in
  ddp)
    echo "Running with DDP"
    srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
      --log_dir='logs' \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$SLURM_GPUS_PER_NODE \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      $(which itwinai) exec-pipeline --config-name config.yaml +pipe_key=training_pipeline strategy=ddp"
    ;;

  deepspeed)
    echo "Running with DeepSpeed"
    srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
      --log_dir='logs_torchrun' \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$SLURM_GPUS_PER_NODE \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      $(which itwinai) exec-pipeline --config-name config.yaml +pipe_key=training_pipeline \
      strategy=deepspeed"
    ;;

  horovod)
    echo "Running with Horovod"
    srun --cpu-bind=none \
      --ntasks-per-node=$SLURM_GPUS_PER_NODE \
      --cpus-per-task=$SLURM_CPUS_PER_TASK \
      --ntasks=$((SLURM_NNODES * SLURM_GPUS_PER_NODE)) \
      $(which itwinai) exec-pipeline --config-name config.yaml +pipe_key=training_pipeline \
      strategy=horovod
    ;;

  *)
    echo "Error: Unknown strategy '$STRATEGY' in config.yaml"
    exit 1
    ;;
esac