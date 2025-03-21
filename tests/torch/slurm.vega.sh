#!/bin/bash

# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

# shellcheck disable=all

# SLURM jobscript for Vega systems

# Job configuration
#SBATCH --job-name=3dgan_training
#SBATCH --account=s24r05-03-users
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:20:00

# Resources allocation
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
# SBATCH --mem-per-gpu=10G
# SBATCH --exclusive

echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: CUDA_VISIBLE_DEVICES (before): $CUDA_VISIBLE_DEVICES"

# Load environment modules
ml --force purge
ml Python/3.11.5-GCCcore-13.2.0 
ml CMake/3.24.3-GCCcore-11.3.0
ml mpi4py
ml OpenMPI
ml CUDA/12.3
ml GCCcore/11.3.0
ml NCCL
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0 # this is needed by horovod!
module unload OpenSSL

# source ~/.bashrc
# source ../../.venv-pytorch/bin/activate

# Setup env for distributed ML
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_PER_NODE - 1)))
echo "DEBUG: CUDA_VISIBLE_DEVICES (after): $CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=1
if [ $(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE)) -gt 0 ] ; then
  export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))
fi

# Adjust itwinai logging level to help with debugging 
export ITWINAI_LOG_LEVEL=DEBUG
# Disable ANSI colors in log files
export NO_COLOR=1

export NCCL_SOCKET_IFNAME=ib0   # Use infiniband interface ib0
export NCCL_DEBUG=INFO          # Enables detailed logging
export NCCL_P2P_DISABLE=0       # Ensure P2P communication is enabled
export NCCL_IB_DISABLE=0        # Ensure InfiniBand is used if available
export GLOO_SOCKET_IFNAME=ib0   # Ensure GLOO (fallback) also uses the correct interface

# Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages inside the container
# https://docs.sylabs.io/guides/4.1/user-guide/environment_and_metadata.html#environment-from-the-host
unset PYTHONPATH

# Launch distributed job in container with torchrun
torchrun_launcher ()
{
  # Stop Ray processes, if any
  srun singularity exec --nv $CONTAINER_PATH ray stop

  # --no-python is needed when running commands which are not python scripts (e.g., pytest, itwinai)
  # --redirects=\$(((SLURM_NODEID)) && echo "3" || echo "1:3,2:3,3:3"): redirect stdout and stderr to 
  # torchrun logs dir for workers having rank !=0 
  srun --cpu-bind=none --ntasks-per-node=1 \
    singularity exec --nv $CONTAINER_PATH /bin/bash -c "torchrun \
    --log_dir='logs_torchrun' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --node-rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):29500 \
    --no-python \
    --redirects=\$(((SLURM_NODEID)) && echo "3" || echo "1:3,2:3,3:3") \
    ${1}"
}

# Launch distribtued job in container with mpirun
mpirun_launcher ()
{
  # Stop Ray processes, if any
  srun singularity exec --nv $CONTAINER_PATH ray stop

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  export UCX_TLS=self,sm,rc,ud
  export OMPI_MCA_PML="ucx"
  export OMPI_MCA_osc="ucx"

  # Get the node list from Slurm
  NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
  SLOTS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

  # Create the string for horovodrun format, e.g., "node1:4,node2:4,..."
  HOSTFILE=""
  for NODE in $NODELIST; do
      if [ -z "$HOSTFILE" ]; then
          # First node, no comma
          HOSTFILE="$NODE:$SLOTS_PER_NODE"
      else
          # Subsequent nodes, prepend comma
          HOSTFILE="$HOSTFILE,$NODE:$SLOTS_PER_NODE"
      fi
  done

  # Display the generated hostfile (optional)
  echo "Generated host string: $HOSTFILE"

  # Calculate the total number of processes (GPUs in this case)
  TOTAL_PROCESSES=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))

  # Calculate the total number of processes (GPUs in this case)
  echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
  echo "SLURM_NNODES: $SLURM_NNODES"
  echo "TOTAL_MPI_PROCESSES: $TOTAL_PROCESSES"
  echo "SLURM_CPUS_PER_GPU: $SLURM_CPUS_PER_GPU"
  echo

  # # Get OpenMPI installation prefixes (locally and in container)
  # OMPI_CONTAINER="$(singularity exec ${CONTAINER_PATH} /bin/bash -c 'ompi_info' | grep Prefix | awk '{ print $2 }')"
  # OMPI_HOST="$(ompi_info | grep Prefix | awk '{ print $2 }')"
  # # If you want to explicitly mount host OpenMPI in container use --bind "${OMPI_HOST}":"${OMPI_CONTAINER}"  

  # Create mpirun logs folder
  mkdir -p "logs_mpirun/$SLURM_JOB_ID"

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  # "if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > "logs_mpirun/$SLURM_JOB_ID/rank.$OMPI_COMM_WORLD_RANK" 2>&1; fi; exec" redirects stdout and stderr of ranks != 0
  # Logs of the main woker (rank == 0) will be incorportated into the standard SLURM out and err files
  mpirun -H "${HOSTFILE}" -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core \
    singularity exec --nv  \
    "${CONTAINER_PATH}" /bin/bash -c \
    'echo "Rank: $OMPI_COMM_WORLD_RANK, lrank: $OMPI_COMM_WORLD_LOCAL_RANK, Size: $OMPI_COMM_WORLD_SIZE, LD_LIBRARY_PATH=$LD_LIBRARY_PATH" &&  \
    if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > "logs_mpirun/$SLURM_JOB_ID/rank.$OMPI_COMM_WORLD_RANK" 2>&1; fi; exec '"${1}"
}

# Launch distribtued job in container with srun
srun_launcher ()
{

  # Stop Ray processes, if any
  srun singularity exec --nv $CONTAINER_PATH ray stop

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  export UCX_TLS=self,sm,rc,ud
  export OMPI_MCA_PML="ucx"
  export OMPI_MCA_osc="ucx"

  # This tells UCX to enable fork safety when using RDMA (InfiniBand)
  export RDMAV_FORK_SAFE=1

  # Create mpirun logs folder
  mkdir -p "logs_srun/$SLURM_JOB_ID"

  # # Get OpenMPI installation prefixes (locally and in container)
  # OMPI_CONTAINER="$(singularity exec ${CONTAINER_PATH} /bin/bash -c 'ompi_info' | grep Prefix | awk '{ print $2 }')"
  # OMPI_HOST="$(ompi_info | grep Prefix | awk '{ print $2 }')"
  # # If you want to explicitly mount host OpenMPI in container use --bind "${OMPI_HOST}":"${OMPI_CONTAINER}"  
  
  # "if [ $SLURM_PROCID  -ne 0 ]; then exec > "logs_srun/$SLURM_JOB_ID/rank.$SLURM_PROCID" 2>&1; fi; exec" redirects stdout and stderr of ranks != 0
  # Logs of the main woker (rank == 0) will be incorportated into the standard SLURM out and err files
  srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE \
    --cpus-per-task=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE)) \
    --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
    singularity exec --nv \
    "${CONTAINER_PATH}" /bin/bash -c \
    'echo "Rank: $SLURM_PROCID, LD_LIBRARY_PATH=$LD_LIBRARY_PATH" && \
    if [ $SLURM_PROCID  -ne 0 ]; then exec > "logs_srun/$SLURM_JOB_ID/rank.$SLURM_PROCID" 2>&1; fi; exec '"${1}"
}

# Launch distribtued job in container with Ray
ray_launcher ()
{

  # Remove ray metadata if present
  srun rm -rf /tmp/ray & disown

  # This tells Tune to not change the working directory to the trial directory
  # which makes relative paths accessible from inside a trial
  export RAY_CHDIR_TO_TRIAL_DIR=0
  export RAY_DEDUP_LOGS=0
  export RAY_USAGE_STATS_DISABLE=1

  # Disable colors in output
  export RAY_COLOR_PREFIX=0

  #########   Set up Ray cluster   ########

  # Get the node names
  nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  mapfile -t nodes_array <<< "$nodes"
  echo "Nodes in nodes_array: ${nodes_array[@]}"

  # The head node will act as the central manager (head) of the Ray cluster.
  head_node=${nodes_array[0]}
  port=7639       # This port will be used by Ray to communicate with worker nodes.

  echo "Starting HEAD at $head_node"
  # Start Ray on the head node.
  # The `--head` option specifies that this node will be the head of the Ray cluster.
  # `srun` submits a job that runs on the head node to start the Ray head with the specified 
  # number of CPUs and GPUs.

  srun --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv $CONTAINER_PATH  \
      ray start \
      --head \
      --log-color false \
      --node-ip-address="$head_node" \
      --port=$port \
      --num-cpus "$SLURM_CPUS_PER_TASK" \
      --num-gpus "$SLURM_GPUS_PER_NODE" \
      --block &

  # Wait for a few seconds to ensure that the head node has fully initialized.
  sleep 15

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
        singularity exec --nv $CONTAINER_PATH \
          ray start \
          --address "$head_node":"$port" \
          --log-color false \
          --redis-password='5241580000000000' \
          --num-cpus "$SLURM_CPUS_PER_TASK" \
          --num-gpus "$SLURM_GPUS_PER_NODE" \
          --block &
      
      sleep 15 # Wait before starting the next worker to prevent race conditions.
  done
  echo All Ray workers started.

  # Check cluster
  singularity exec --nv $CONTAINER_PATH ray status
  echo "============================================="

  # Run command without srun
  singularity exec --nv $CONTAINER_PATH $1

}

# Dual echo on both stdout and stderr
decho ()
{
  echo "$@"
  >&2 echo "$@"
}


######################   Initial checks   ######################

# Env vairables check
if [ -z "$DIST_MODE" ]; then 
  >&2 echo "ERROR: env variable DIST_MODE is not set. Allowed values are 'horovod', 'ddp' or 'deepspeed'"
  exit 1
fi
if [ -z "$RUN_NAME" ]; then 
  >&2 echo "WARNING: env variable RUN_NAME is not set. It's a way to identify some specific run of an experiment."
  RUN_NAME=$DIST_MODE
fi
if [ -z "$COMMAND" ]; then 
  >&2 echo "ERROR: env variable COMMAND is not set. It's the python command to execute."
  exit 1
fi
if [ -z "$CONTAINER_PATH" ]; then 
  >&2 echo "WARNING: env variable CONTAINER_PATH is not set. It's the path to a singularity container."
  exit 1
fi

# OpenMPI version
HOST_OMPI_V="$(ompi_info --parsable | grep ompi:version:full: |  cut -d':' -f4 | cut -d'.' -f1,2)"
CONTAINER_OMPI_V="$(singularity exec $CONTAINER_PATH ompi_info --parsable | grep ompi:version:full: |  cut -d':' -f4 | cut -d'.' -f1,2)"

if [ "$HOST_OMPI_V" != "$CONTAINER_OMPI_V" ]; then
  >&2 echo "ERROR: Host OpenMPI minor version ($HOST_OMPI_V) does not match with container's OpenMPI minor version ($CONTAINER_OMPI_V). This may cause problems." 
  # exit 1
fi
echo -e "\nHost and container's OpenMPI minor versions match: ($HOST_OMPI_V) - ($CONTAINER_OMPI_V)\n" 

# Get GPUs info per node
srun --cpu-bind=none --ntasks-per-node=1 bash -c 'echo -e "NODE hostname: $(hostname)\n$(nvidia-smi)\n\n"'

# Print env variables
echo "RUN_NAME: $RUN_NAME"
echo "DIST_MODE: $DIST_MODE"
echo "CONTAINER_PATH: $CONTAINER_PATH"
echo "COMMAND: $COMMAND"

######################   Execute command   ######################

if [ "${DIST_MODE}" == "ddp" ] ; then

  decho -e "\nLaunching DDP strategy with torchrun"
  torchrun_launcher "${COMMAND}"

elif [ "${DIST_MODE}" == "deepspeed" ] ; then

  decho -e "\nLaunching DeepSpeed strategy with torchrun"
  torchrun_launcher "${COMMAND}"

  decho -e "\nLaunching DeepSpeed strategy with mpirun"
  mpirun_launcher "python -m ${COMMAND}"

  decho -e "\nLaunching DeepSpeed strategy with srun"
  srun_launcher "python -m ${COMMAND}"

elif [ "${DIST_MODE}" == "horovod" ] ; then

  decho -e "\nLaunching Horovod strategy with mpirun"
  mpirun_launcher "python -m ${COMMAND}"

  decho -e "\nLaunching Horovod strategy with srun"
  srun_launcher "python -m ${COMMAND}"

elif [ "${DIST_MODE}" == "ray" ] ; then

  decho -e "\nLaunching Ray tests"
  ray_launcher "${COMMAND}"

else
  >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
  exit 1
fi
