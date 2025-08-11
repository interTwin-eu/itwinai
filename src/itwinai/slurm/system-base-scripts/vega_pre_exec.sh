echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: CUDA_VISIBLE_DEVICES (before): $CUDA_VISIBLE_DEVICES"
echo "DEBUG: DIST_MODE: $DIST_MODE"
echo

# Load environment modules
ml --force purge
ml CMake/3.29.3-GCCcore-13.3.0
ml mpi4py/3.1.5
ml OpenMPI/4.1.6-GCC-13.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml CUDA/12.6.0
ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
ml Python/3.12.3-GCCcore-13.3.0

# Setup env for distributed ML
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
export CUDA_VISIBLE_DEVICES
echo "DEBUG: CUDA_VISIBLE_DEVICES (after): $CUDA_VISIBLE_DEVICES"

OMP_NUM_THREADS=1
if [ $((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) -gt 0 ] ; then
  OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))
fi
export OMP_NUM_THREADS

# Adjust itwinai logging level to help with debugging 
export ITWINAI_LOG_LEVEL=DEBUG
# Disable ANSI colors in log files
export NO_COLOR=1

export NCCL_SOCKET_IFNAME=ib0   # Use infiniband interface ib0
export NCCL_P2P_DISABLE=0       # Ensure P2P communication is enabled
export NCCL_IB_DISABLE=0        # Ensure InfiniBand is used if available
export GLOO_SOCKET_IFNAME=ib0   # Ensure GLOO (fallback) also uses the correct interface

# Debug network setup and NCCL (uncomment to debug)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export FI_LOG_LEVEL=info
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
export HYDRA_FULL_ERROR=1

# Launchers
function torchrun-launcher(){
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  MASTER_PORT=54123
  export MASTER_ADDR MASTER_PORT
  torchrun_command='torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_conf=is_host="$(( SLURM_NODEID == 0 ? 1 : 0 ))" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --redirects="$(((SLURM_NODEID)) && echo "3" || echo "1:3,2:3,3:3")" \
    --no-python'
  torchrun_command="$torchrun_command $1"
  srun --cpu-bind=none --ntasks-per-node=1 bash -c "$torchrun_command"
}

function torchrun-launcher-container(){
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  MASTER_PORT=54123
  export MASTER_ADDR MASTER_PORT
  torchrun_command='torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_conf=is_host="$(( SLURM_NODEID == 0 ? 1 : 0 ))" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --redirects="$(((SLURM_NODEID)) && echo "3" || echo "1:3,2:3,3:3")" \
    --no-python'
  torchrun_command="$torchrun_command $1"
  srun --cpu-bind=none --ntasks-per-node=1 "$CONTAINER_PATH" /bin/bash -c "$torchrun_command"
}

function py-spy-torchrun-launcher(){
  local rate="$1"
  local output="$2"

  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  MASTER_PORT=54123
  export MASTER_ADDR MASTER_PORT

  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c 'py-spy record -r "$0" -s -o "$1" -f raw -- torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_JOB_NUM_NODES" \
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

  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  MASTER_PORT=54123
  export MASTER_ADDR MASTER_PORT

  srun --cpu-bind=none --ntasks-per-node=1 \
    "$CONTAINER_PATH" /bin/bash -c 'py-spy record -r "$0" -s -o "$1" -f raw -- torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_conf=is_host="$(( SLURM_NODEID == 0 ? 1 : 0 ))" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    --no-python \
    "$@"' "$rate" "$output" "$3"
}


function srun-launcher (){
  # Create mpirun logs folder
  mkdir -p "logs_srun/$SLURM_JOB_ID"

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  export UCX_TLS=self,sm,rc,ud
  export OMPI_MCA_PML="ucx"
  export OMPI_MCA_osc="ucx"

  # This tells UCX to enable fork safety when using RDMA (InfiniBand)
  export RDMAV_FORK_SAFE=1

  # Launch command
  srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node="$SLURM_GPUS_PER_NODE" \
      --cpus-per-task=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) \
      --ntasks=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES)) \
      /bin/bash -c \
      'if [ $SLURM_PROCID  -ne 0 ]; then exec > "logs_srun/$SLURM_JOB_ID/rank.$SLURM_PROCID" 2>&1; fi; exec '"${1}"
}

function srun-launcher-container (){
  # Create mpirun logs folder
  mkdir -p "logs_srun/$SLURM_JOB_ID"

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  export UCX_TLS=self,sm,rc,ud
  export OMPI_MCA_PML="ucx"
  export OMPI_MCA_osc="ucx"

  # This tells UCX to enable fork safety when using RDMA (InfiniBand)
  export RDMAV_FORK_SAFE=1

  # Launch command
  srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node="$SLURM_GPUS_PER_NODE" \
      --cpus-per-task=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE)) \
      --ntasks=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES)) \
      "$CONTAINER_PATH" /bin/bash -c \
      'if [ $SLURM_PROCID  -ne 0 ]; then exec > "logs_srun/$SLURM_JOB_ID/rank.$SLURM_PROCID" 2>&1; fi; exec '"${1}"
}

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
        ray start --head --node-ip-address="$head_node" --port=$port \
        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE"  --block &

    # Wait for a few seconds to ensure that the head node has fully initialized.
    sleep 10

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
            ray start --address "$head_node":"$port" --redis-password='5241580000000000' \
            --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &
        
        sleep 10 # Wait before starting the next worker to prevent race conditions.
    done
    echo All Ray workers started.
    sleep 30

    # Run command without srun
    bash -c "$1" 
}

function ray-launcher-container(){

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
        ray start --head --node-ip-address="$head_node" --port=$port \
        --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE"  --block &

    # Wait for a few seconds to ensure that the head node has fully initialized.
    sleep 10

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
            ray start --address "$head_node":"$port" --redis-password='5241580000000000' \
            --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &
        
        sleep 10 # Wait before starting the next worker to prevent race conditions.
    done
    echo All Ray workers started.
    sleep 30

    # Run command without srun
    "$CONTAINER_PATH" /bin/bash -c "$1" 
}
