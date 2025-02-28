#!/bin/bash

# SLURM jobscript for JSC systems

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=intertwin
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=01:00:00

# Resources allocation
#SBATCH --partition=develbooster
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# Load environment modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Job info
echo "DEBUG: TIME: $(date)"
sysN="$(uname -n | cut -f2- -d.)"
sysN="${sysN%%[0-9]*}"
echo "Running on system: $sysN"
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
if [ "$DEBUG" = true ] ; then
  echo "DEBUG: NCCL_DEBUG=INFO" 
  export NCCL_DEBUG=INFO
fi
echo

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

export NCCL_SOCKET_IFNAME=ib0   # Use infiniband interface ib0
export NCCL_DEBUG=INFO          # Enables detailed logging
export NCCL_P2P_DISABLE=0       # Ensure P2P communication is enabled
export NCCL_IB_DISABLE=0        # Ensure InfiniBand is used if available
export GLOO_SOCKET_IFNAME=ib0   # Ensure GLOO (fallback) also uses the correct interface


torchrun_launcher(){
  
  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $1"
}


# Launch atmorep. Ref: https://github.com/clessig/atmorep/blob/iluise/develop/itwinai/slurm_atmorep.sh
torchrun_launcher debug_trainer.py

