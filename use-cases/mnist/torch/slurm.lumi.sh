#!/bin/bash

# SLURM jobscript for Lumi systems

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=project_465001592
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00

# Resources allocation
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

set -e

# Load environment modules
ml LUMI partition/G rocm/6.2.2

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
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: TRAINING_CMD: $TRAINING_CMD"
echo

# Optional: Inject the environment variables for NCCL debugging into the container.   
# This will produce a lot of debug output!     
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Make sure GPUs are up
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# Set ROCR_VISIBLE_DEVICES so that each task uses the proper GPU
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

# Report affinity to check
echo "Rank $SLURM_PROCID --> $(taskset -p $$); GPU $ROCR_VISIBLE_DEVICES"


# Setup env for distributed ML
# export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
# echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=1
if [ "$SLURM_GPUS_PER_NODE" -gt 0 ] ; then
  export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))
fi

export HYDRA_FULL_ERROR=1

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
if [ -z "$CONTAINER_PATH" ]; then 
  >&2 echo "WARNING: env variable CONTAINER_PATH is not set."
fi

function ray-launcher(){
  num_gpus=$SLURM_GPUS_PER_NODE
  num_cpus=$SLURM_CPUS_PER_TASK

  # This tells Tune to not change the working directory to the trial directory
  # which makes relative paths accessible from inside a trial
  export RAY_CHDIR_TO_TRIAL_DIR=0
  export RAY_DEDUP_LOGS=0
  export RAY_USAGE_STATS_DISABLE=1

  # Disable colors in output
  export NO_COLOR=1
  export RAY_COLOR_PREFIX=0

  # Fix (?) for: HIP error: invalid device ordinal
  export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
  export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  #########   Set up Ray cluster   ########

  # Get the node names
  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  mapfile -t nodes_array <<< "$nodes"

  # The head node will act as the central manager (head) of the Ray cluster.
  head_node=${nodes_array[0]}
  port=7639       # This port will be used by Ray to communicate with worker nodes.

  echo "Starting HEAD at $head_node"
  # Start Ray on the head node.
  # The `--head` option specifies that this node will be the head of the Ray cluster.
  # `srun` submits a job that runs on the head node to start the Ray head with the specified 
  # number of CPUs and GPUs.
  srun --nodes=1 --ntasks=1 -w "$head_node" \
      singularity exec --bind $(pwd):$(pwd) --rocm $CONTAINER_PATH  bash -c " \
      source /opt/miniconda3/bin/activate pytorch && \
      ray start \
        --head \
        --node-ip-address="$head_node" \
        --port=$port \
      --num-cpus "$num_cpus" \
      --num-gpus "$num_gpus"  \
      --log-color false \
      --block" &

  # Wait for a few seconds to ensure that the head node has fully initialized.
  sleep 5

  echo HEAD node started.

  # Start Ray worker nodes
  # These nodes will connect to the head node and become part of the Ray cluster.
  worker_num=$((SLURM_JOB_NUM_NODES - 1))    # Total number of worker nodes (excl the head node)
  for ((i = 1; i <= worker_num; i++)); do
      node_i=${nodes_array[$i]}   # Get the current worker node hostname.
      echo "Starting WORKER $i at $node_i"

      # Use srun to start Ray on the worker node and connect it to the head node.
      # The `--address` option tells the worker node where to find the head node.
      srun --nodes=1 --ntasks=1 -w "$node_i" \
        singularity exec --bind $(pwd):$(pwd) --rocm $CONTAINER_PATH bash -c " \
        source /opt/miniconda3/bin/activate pytorch && \
        ray start \
          --address "$head_node":"$port" \
          --redis-password='5241580000000000' \
          --num-cpus "$num_cpus" \
          --num-gpus "$num_gpus" \
          --log-color false \
          --block" &
      
      sleep 5 # Wait before starting the next worker to prevent race conditions.
  done
  sleep 20
  echo All Ray workers started.

  # Check cluster
  singularity exec --rocm --bind $(pwd):$(pwd) $CONTAINER_PATH bash -c "\
    source /opt/miniconda3/bin/activate pytorch && \
    ray status"
  echo "============================================="

  singularity exec --rocm --bind $(pwd):$(pwd) $CONTAINER_PATH bash -c '\
    source /opt/miniconda3/bin/activate pytorch && \
    echo HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES'

  # Run command without srun
  singularity exec --bind $(pwd):$(pwd) $CONTAINER_PATH bash -c "\
    source /opt/miniconda3/bin/activate pytorch && \
    $1"
}

function torchrun-launcher(){
  srun --cpu-bind=none --ntasks-per-node=1 \
    singularity exec --rocm --bind $(pwd):$(pwd) $CONTAINER_PATH /bin/bash -c "\
    source /opt/miniconda3/bin/activate pytorch && \
    torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node-rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
    --no-python \
    --redirects=\$(((SLURM_NODEID)) && echo "3" || echo "1:3,2:3,3:3") \
    $1"
}

function srun-launcher(){
  srun --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE \
    --cpus-per-task=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE)) \
    --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
    $1
}

# Dual echo on both stdout and stderr
function decho (){
  echo "$@"
  >&2 echo "$@"
}

function separation(){

  decho
  decho "======================================================================================"
  decho "======================================================================================"
  decho "======================================================================================"
  decho

}

# Get GPUs info per node
srun --cpu-bind=none --ntasks-per-node=1 bash -c 'echo -e "NODE hostname: $(hostname)\n$(rocm-smi)\n\n"'

export ITWINAI_LOG_LEVEL="DEBUG"

# Launch training
if [ "$DIST_MODE" == "ddp" ] ; then
  echo "DDP training: $TRAINING_CMD"
  torchrun-launcher "$TRAINING_CMD"

  separation

  ray-launcher "$TRAINING_CMD"
  
elif [ "$DIST_MODE" == "deepspeed" ] ; then
  echo "DEEPSPEED training: $TRAINING_CMD"
  torchrun-launcher "$TRAINING_CMD"

  separation

  ray-launcher "$TRAINING_CMD"

elif [ "$DIST_MODE" == "horovod" ] ; then
  echo "HOROVOD training: $TRAINING_CMD"
  srun-launcher "$TRAINING_CMD"

  separation

  ray-launcher "$TRAINING_CMD"
else
  >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
  exit 1
fi
