#!/bin/bash

# SLURM jobscript for Lumi systems

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=eehpc-dev-2024d11-012a
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00

# Resources allocation
#SBATCH --partition=normal-arm
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=128G


set -e

# # Load environment modules
# ml --force purge


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
echo "DEBUG: NWORKERS_PER_NODE: $NWORKERS_PER_NODE"
echo

export HYDRA_FULL_ERROR=1
export ITWINAI_LOG_LEVEL="DEBUG"

# Env variables check
if [ -z "$DIST_MODE" ]; then 
  >&2 echo "ERROR: env variable DIST_MODE is not set. Allowed values are 'horovod', 'ddp' or 'deepspeed'"
  exit 1
fi
if [ -z "$TRAINING_CMD" ]; then 
  >&2 echo "ERROR: env variable TRAINING_CMD is not set. It's the python command to execute."
  exit 1
fi
if [ -z "$CONTAINER_PATH" ]; then 
  >&2 echo "WARNING: env variable CONTAINER_PATH is not set."
  exit 1
fi
if [ -z "$NWORKERS_PER_NODE" ]; then 
  >&2 echo "WARNING: env variable NWORKERS_PER_NODE is not set."
  exit 1
fi


function ray-launcher(){
  
  # This tells Tune to not change the working directory to the trial directory
  # which makes relative paths accessible from inside a trial
  export RAY_CHDIR_TO_TRIAL_DIR=0
  export RAY_DEDUP_LOGS=0
  export RAY_USAGE_STATS_DISABLE=1

  # Disable colors in output
  export NO_COLOR=1
  export RAY_COLOR_PREFIX=0

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
  singularity exec \
    --bind "$(pwd)" \
    --bind $ITWINAI_DYNAMIC_BIND \
    "$CONTAINER_PATH" bash -c " \
      ray start \
        --head \
        --node-ip-address=$head_node \
        --port=$port \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_PER_NODE \
        --log-color false \
        --block" &
  # Wait for a few seconds to ensure that the head node has fully initialized.
  sleep 8

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
      singularity exec \
      --bind "$(pwd)" \
      --bind $ITWINAI_DYNAMIC_BIND \
      "$CONTAINER_PATH" bash -c " \
      ray start \
        --address $head_node:$port \
        --redis-password='5241580000000000' \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_PER_NODE \
        --log-color false \
        --block" &      
      sleep 8 # wait before starting the next worker to prevent race conditions.
  done
  sleep 30
  echo all ray workers started.

  # Check cluster
  singularity exec $CONTAINER_PATH bash -c "ray status"
  echo "============================================="

  # Run command without srun
  singularity exec \
    --bind $(pwd) \
    --bind $ITWINAI_DYNAMIC_BIND \
    $CONTAINER_PATH bash -c "$1"
}

function torchrun-launcher(){
  srun --cpu-bind=none --ntasks-per-node=1 \
    singularity exec \
      --bind $(pwd) \
      --bind $ITWINAI_DYNAMIC_BIND \
      $CONTAINER_PATH /bin/bash -c "\
        torchrun \
        --log_dir='logs_torchrun' \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$NWORKERS_PER_NODE \
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
  # May be superfluous and therefor not needed
  # - OMP_NUM_THREADS is already set earlier based on GPU count
  # - MPICH_GPU_SUPPORT_ENABLED is not required unless using GPU-aware MPI with MPICH.
  # Keeping them has no harmful effect but may be redundant
  export OMP_NUM_THREADS=1
  export MPICH_GPU_SUPPORT_ENABLED=1

  srun --cpu-bind=none \
    --mpi=pmi2 \
    --ntasks-per-node=$NWORKERS_PER_NODE \
    --cpus-per-task=$(($SLURM_CPUS_PER_TASK / $NWORKERS_PER_NODE)) \
    --ntasks=$(($NWORKERS_PER_NODE * $SLURM_NNODES)) \
    singularity exec \
    --bind $(pwd) \
    --bind $ITWINAI_DYNAMIC_BIND \
    $CONTAINER_PATH /bin/bash -c "$1"
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

  # Horovod is not supported anymore by Ray

else
  >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
  exit 1
fi
