#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# SLURM job script for JUWELS system

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=intertwin
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:25:00

# Resources allocation
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive

# Load environment modules
ml --force purge
ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git

# Job info
echo "DEBUG: TIME: $(date)"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "DEBUG: TRAINING_CMD: $TRAINING_CMD"

# Setup env for distributed ML
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
export CUDA_VISIBLE_DEVICES
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=1
if [ "$SLURM_GPUS_PER_NODE" -gt 0 ] ; then
    export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))
fi

# Setup network sockets (prefer infiniband)
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0


# Env variables check
if [ -z "$DIST_MODE" ]; then 
    >&2 echo "ERROR: env variable DIST_MODE is not set. \
        Allowed values are 'horovod', 'ddp' or 'deepspeed'"
    exit 1
fi
if [ -z "$TRAINING_CMD" ]; then 
    >&2 echo "ERROR: env variable TRAINING_CMD is not set. \
        It's the command to execute."
    exit 1
fi

function torchrun-launcher-container(){
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"
    MASTER_PORT=54123
    export MASTER_ADDR MASTER_PORT

    torchrun_command='torchrun_jsc \
        --log_dir="logs_torchrun" \
        --nnodes="$SLURM_NNODES" \
        --nproc_per_node="$SLURM_GPUS_PER_NODE" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_conf=is_host="$(( SLURM_NODEID == 0 ? 1 : 0 ))" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
        --no-python'
    torchrun_command="$torchrun_command $1"

    srun --cpu-bind=none --ntasks-per-node=1 \
        singularity exec --nv -B "$PWD/data":/data -B "$PWD":/experiments \
        "$CONTAINER_PATH" /bin/bash -c "$torchrun_command"
}

function srun-launcher-container(){
    srun --cpu-bind=none --ntasks-per-node="$SLURM_GPUS_PER_NODE" \
        --cpus-per-task=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) \
        --ntasks=$((SLURM_GPUS_PER_NODE * SLURM_NNODES)) \
            singularity exec --nv -B "$PWD/data":/data -B "$PWD":/experiments \
            "$CONTAINER_PATH" /bin/bash -c "$1"
}

# Launch training

# export ITWINAI_LOG_LEVEL="DEBUG"

if [ "$DIST_MODE" == "ddp" ] ; then
    echo "DDP training: $TRAINING_CMD"
    torchrun-launcher-container "$TRAINING_CMD"
elif [ "$DIST_MODE" == "deepspeed" ] ; then
    echo "DEEPSPEED training: $TRAINING_CMD"
    torchrun-launcher-container "$TRAINING_CMD"
elif [ "$DIST_MODE" == "horovod" ] ; then
    echo "HOROVOD training: $TRAINING_CMD"
    srun-launcher-container "$TRAINING_CMD"
else
    >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
    exit 1
fi