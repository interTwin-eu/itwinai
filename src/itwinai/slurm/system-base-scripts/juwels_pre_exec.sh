# Load environment modules
ml --force purge
ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git

# Job info
echo "DEBUG: TIME: $(date)"
echo "Running on system: $SYSTEMNAME"
echo "DEBUG: EXECUTE: $EXEC"
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

if [ "$DEBUG" = true ] ; then
  echo "DEBUG: NCCL_DEBUG=INFO" 
  export NCCL_DEBUG=INFO
fi
echo

# Setup env for distributed ML
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
export CUDA_VISIBLE_DEVICES
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

OMP_NUM_THREADS=1
if [ "$SLURM_GPUS_PER_NODE" -gt 0 ] ; then
  OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))
fi
export OMP_NUM_THREADS


# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0


function torchrun-launcher(){
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
  srun --cpu-bind=none --ntasks-per-node=1 bash -c "$torchrun_command"
}

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
  srun --cpu-bind=none --ntasks-per-node=1 "$CONTAINER_PATH" /bin/bash -c "$torchrun_command"
}

function py-spy-torchrun-launcher(){
  local rate="$1"
  local output="$2"

  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"
  MASTER_PORT=54123
  export MASTER_ADDR MASTER_PORT

  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c 'py-spy record -r "$0" -s -o "$1" -f raw -- torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_conf=is_host="$(( SLURM_NODEID == 0 ? 1 : 0 ))" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --no-python \
    "$@"' "$rate" "$output" "$3"
}

function py-spy-torchrun-launcher-container(){
  local rate="$1"
  local output="$2"

  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"
  MASTER_PORT=54123
  export MASTER_ADDR MASTER_PORT

  srun --cpu-bind=none --ntasks-per-node=1 \
    "$CONTAINER_PATH" /bin/bash -c 'py-spy record -r "$0" -s -o "$1" -f raw -- torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_conf=is_host="$(( SLURM_NODEID == 0 ? 1 : 0 ))" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --no-python \
    "$@"' "$rate" "$output" "$3"
}

function srun-launcher(){
  srun --cpu-bind=none --ntasks-per-node="$SLURM_GPUS_PER_NODE" \
    --cpus-per-task=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) \
    --ntasks=$((SLURM_GPUS_PER_NODE * SLURM_NNODES)) \
    bash -c "$1"
}

# Get GPUs info per node
srun --cpu-bind=none --ntasks-per-node=1 \
  bash -c 'echo -e "NODE hostname: $(hostname)\n$(nvidia-smi)\n\n"'

export ITWINAI_LOG_LEVEL="DEBUG"

function srun-launcher-container(){
  srun --cpu-bind=none --ntasks-per-node="$SLURM_GPUS_PER_NODE" \
    --cpus-per-task=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) \
    --ntasks=$((SLURM_GPUS_PER_NODE * SLURM_NNODES)) \
      "$CONTAINER_PATH" /bin/bash -c "$1"
}

# Get GPUs info per node
srun --cpu-bind=none --ntasks-per-node=1 \
  bash -c 'echo -e "NODE hostname: $(hostname)\n$(nvidia-smi)\n\n"'

export ITWINAI_LOG_LEVEL="DEBUG"

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
      ray start --head --node-ip-address="$head_node"i --port=$port \
      --num-cpus "$num_cpus" --num-gpus "$num_gpus"  --block &

  # Wait for a few seconds to ensure that the head node has fully initialized.
  sleep 1

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
          ray start --address "$head_node"i:"$port" --redis-password='5241580000000000' \
          --num-cpus "$num_cpus" --num-gpus "$num_gpus" --block &
      
      sleep 5 # Wait before starting the next worker to prevent race conditions.
  done
  echo All Ray workers started.

  # Run command without srun
  # if you want the number of workers to be adaptive during distributed training append this:
  # training_pipeline.steps.training_step.ray_scaling_config.num_workers=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))
  $1
}

function ray-launcher-container(){
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
      ray start --head --node-ip-address="$head_node"i --port=$port \
      --num-cpus "$num_cpus" --num-gpus "$num_gpus"  --block &

  # Wait for a few seconds to ensure that the head node has fully initialized.
  sleep 1

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
          ray start --address "$head_node"i:"$port" --redis-password='5241580000000000' \
          --num-cpus "$num_cpus" --num-gpus "$num_gpus" --block &
      
      sleep 5 # Wait before starting the next worker to prevent race conditions.
  done
  echo All Ray workers started.

  # Run command without srun
  # if you want the number of workers to be adaptive during distributed training append this:
  # training_pipeline.steps.training_step.ray_scaling_config.num_workers=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))
  "$CONTAINER_PATH" /bin/bash -c "$1" 
}

