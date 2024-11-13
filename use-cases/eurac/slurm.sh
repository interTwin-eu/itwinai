#!/bin/bash

# SLURM jobscript for JSC systems

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=intertwin
#SBATCH --output=logs_slurm/job.out
#SBATCH --error=logs_slurm/job.err
#SBATCH --time=01:00:00

# Resources allocation
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --exclusive

# Load environment modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# Handling the environment variables
if [ -z "$DEBUG" ]; then 
	DEBUG=false
fi
if [ -z "$DIST_MODE" ]; then 
  >&2 echo "WARNING: env variable DIST_MODE is not set. Defaulting to 'ddp'"
	DIST_MODE=ddp
fi
if [ -z "$RUN_NAME" ]; then 
  >&2 echo "WARNING: env variable RUN_NAME is not set. Defaulting to $DIST_MODE."
  RUN_NAME=$DIST_MODE
fi
if [ -z "$CONFIG_FILE" ]; then 
  CONFIG_FILE=config.yaml
  >&2 echo "WARNING: env variable CONFIG_FILE is not set. Defaulting to $CONFIG_FILE."
fi
if [ -z "$TRAINING_CMD" ]; then 
 	TRAINING_CMD="$(which itwinai) exec-pipeline \
	--config $CONFIG_FILE \
	--pipe-key rnn_training_pipeline \
	-o strategy=$DIST_MODE"
  >&2 echo "WARNING: env variable TRAINING_CMD is not set. Defaulting to $TRAINING_CMD."
fi

if [ -z "$PYTHON_VENV" ]; then 
	PYTHON_VENV="../../envAI_hdfml"
  >&2 echo "WARNING: env variable PYTHON_VENV is not set. Defaulting to $PYTHON_VENV"
fi

# Activating the python venv
source $PYTHON_VENV/bin/activate

# Printing debugging information
if [ "$DEBUG" = true ] ; then
	echo "### DEBUG INFORMATION START ###"
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
  echo "DEBUG: NCCL_DEBUG=INFO" 
	echo "### DEBUG INFORMATION END ###"
  export NCCL_DEBUG=INFO
	export TORCH_DISTRIBUTED_DEBUG="DETAIL"
fi

echo "### SLURM SCRIPT###"
echo "Strategy/Dist. Mode: $DIST_MODE"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Python venv: $PYTHON_VENV "

if [ "$DIST_MODE" = "horovod" ]; then 
	# This is necessary for Horovod as it does not use 
	# torchrun but is MPI-based.
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_GPU" -gt 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
fi

# Preparing the necessary directories
mkdir -p logs_slurm # STDOUT and STDERR for slurm
mkdir -p logs_epoch # Logs used for scalability tests

# Launch training
if [ "$DIST_MODE" == "horovod" ] ; then
	srun --cpu-bind=none \
	--ntasks-per-node=$SLURM_GPUS_PER_NODE \
	--cpus-per-task=$SLURM_CPUS_PER_GPU \
	--ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
	$TRAINING_CMD
else # E.g. for 'deepspeed' or 'ddp'
  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
    $TRAINING_CMD"
fi
