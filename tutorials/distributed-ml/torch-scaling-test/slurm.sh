#!/bin/bash

# SLURM jobscript for JSC systems

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

# Setup env for distributed ML
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_GPU" -gt 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
fi

# Env vairables check
if [ -z "$DIST_MODE" ]; then 
  >&2 echo "ERROR: env variable DIST_MODE is not set. Allowed values are 'horovod', 'ddp' or 'deepspeed'"
  exit 1
fi
if [ -z "$RUN_NAME" ]; then 
  >&2 echo "WARNING: env variable RUN_NAME is not set. It's a way to identify some specific run of an experiment."
  RUN_NAME=$DIST_MODE
fi
if [ -z "$TRAINING_CMD" ]; then 
  >&2 echo "ERROR: env variable TRAINING_CMD is not set. It's the python command to execute."
  exit 1
fi
if [ -z "$PYTHON_VENV" ]; then 
  >&2 echo "WARNING: env variable PYTHON_VENV is not set. It's the path to a python virtual environment."
else
  # Activate Python virtual env
  source $PYTHON_VENV/bin/activate
fi

# Get GPUs info per node
srun --cpu-bind=none --ntasks-per-node=1 bash -c 'echo -e "NODE hostname: $(hostname)\n$(nvidia-smi)\n\n"'

# Launch training
if [ "$DIST_MODE" == "ddp" ] ; then
  echo "DDP training: $TRAINING_CMD"
  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $TRAINING_CMD"
elif [ "$DIST_MODE" == "deepspeed" ] ; then
  echo "DEEPSPEED training: $TRAINING_CMD"
  MASTER_ADDR=$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)i
  export MASTER_ADDR
  export MASTER_PORT=29500 

  srun --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU \
    python -u $TRAINING_CMD --deepspeed

  # # Run with deepspeed launcher: set --ntasks-per-node=1
  # # https://www.deepspeed.ai/getting-started/#multi-node-environment-variables
  # export NCCL_IB_DISABLE=1
  # export NCCL_SOCKET_IFNAME=eth0
  # nodelist=$(scontrol show hostname $SLURM_NODELIST)
  # echo "$nodelist" | sed -e 's/$/ slots=4/' > .hostfile
  # # Requires passwordless SSH access among compute node
  # srun --cpu-bind=none deepspeed --hostfile=.hostfile $TRAINING_CMD --deepspeed
  # rm .hostfile
elif [ "$DIST_MODE" == "horovod" ] ; then
  echo "HOROVOD training: $TRAINING_CMD"
  srun --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU \
    python -u $TRAINING_CMD
else
  >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
  exit 1
fi

