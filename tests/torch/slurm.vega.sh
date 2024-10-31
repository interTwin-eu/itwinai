#!/bin/bash

# SLURM jobscript for Vega systems

# Job configuration
#SBATCH --job-name=3dgan_training
#SBATCH --account=s24r05-03-users
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:10:00

# Resources allocation
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks-per-node=1
# SBATCH --mem-per-gpu=10G
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

ml --force purge
ml Python CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI/4.1.5-GCC-12.3.0 CUDA/12.3
ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0
ml UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0

# ml Python
module unload OpenSSL

source ~/.bashrc

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

  # singularity exec --nv $CONTAINER_PATH /bin/bash -c "torchrun \
  # bash -c "torchrun \
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
    $TRAINING_CMD"

elif [ "$DIST_MODE" == "deepspeed" ] ; then

  # Test deepspeed with torchrun
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
    $TRAINING_CMD"

  # # Test deepspeed with MPIRUN
  # # Get the node list from Slurm
  # NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
  # SLOTS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

  # # Create the string for horovodrun format, e.g., "node1:4,node2:4,..."
  # HOSTFILE=""
  # for NODE in $NODELIST; do
  #     if [ -z "$HOSTFILE" ]; then
  #         # First node, no comma
  #         HOSTFILE="$NODE:$SLOTS_PER_NODE"
  #     else
  #         # Subsequent nodes, prepend comma
  #         HOSTFILE="$HOSTFILE,$NODE:$SLOTS_PER_NODE"
  #     fi
  # done

  # # Display the generated hostfile (optional)
  # echo "Generated host string: $HOSTFILE"

  # # Calculate the total number of processes (GPUs in this case)
  # TOTAL_PROCESSES=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES))

  # # Calculate the total number of processes (GPUs in this case)
  # echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
  # echo "SLURM_NNODES: $SLURM_NNODES"
  # echo "TOTAL_MPI_PROCESSES: $TOTAL_PROCESSES"
  # echo "SLURM_CPUS_PER_GPU: $SLURM_CPUS_PER_GPU"
  # echo

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  export UCX_TLS=self,sm,rc,ud
  export OMPI_MCA_PML="ucx"
  export OMPI_MCA_osc="ucx"

  # echo "Run itwinai's pytest with MPIRUN (no mount)"
  # >&2 echo "Run itwinai's pytest with MPIRUN"
  
  # unset PYTHONPATH # Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages inside the container
  #   #  --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core  \
  #   singularity exec --nv  \
  #   $CONTAINER_PATH /bin/bash -c \
  #   'echo "Rank: $OMPI_COMM_WORLD_RANK, lrank: $OMPI_COMM_WORLD_LOCAL_RANK, Size: $OMPI_COMM_WORLD_SIZE" &&  \
  #   if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'

  echo "Run itwinai's pytest with SRUN (no mount)"
  >&2 echo "Run itwinai's pytest with SRUN"
  unset PYTHONPATH # Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages inside the container
  srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
    singularity exec --nv \
    $CONTAINER_PATH /bin/bash -c \
    'if [ $SLURM_PROCID  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'

elif [ "$DIST_MODE" == "horovod" ] ; then
  # echo "Horovod is not currently supported in conjuction with containers"
  # exit 2

  echo "HOROVOD training: $TRAINING_CMD"

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

  # https://doc.vega.izum.si/mpi/#multi-node-jobs
  export UCX_TLS=self,sm,rc,ud
  export OMPI_MCA_PML="ucx"
  export OMPI_MCA_osc="ucx"


  # # This fails because it does not find enough slots
  # horovodrun -H $HOSTFILE -np $TOTAL_PROCESSES \
  #   python -m pytest -vs -m mpirun test_distribtued.py

  # # It works
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe \
  #   python -m pytest -vs -m mpirun test_distribtued.py
  
  # # It works and allows to suppress output from workers having rank != 0
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe \
  #   bash -c 'if [ $OMPI_COMM_WORLD_RANK -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m pytest -vs -m mpirun '
  
  # # It works as well
  # srun --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
  #   python -m pytest -vs -m mpirun test_distribtued.py

  # # It works as well and suppresses output
  # srun --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
  #   bash -c 'if [ $SLURM_PROCID  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'

  # echo "Test MPI setup on mpibench using SRUN (no container)"
  # >&2 echo "Test MPI setup on mpibench using SRUN"
  # srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
  #   ./mpibench


  #   >&2 echo "Test MPI setup on mpibench using MPIRUN (no container)"
  #  echo "Test MPI setup on mpibench using MPIRUN"
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core \
  #   ./mpibench

  #############################################
  # Container

  # # TODO: try to remove this block from the web...
  # export OMPI_MCA_btl="^openib"
  # export OMPI_MCA_orte_keep_fqdn_hostnames="true"

  OMPI_CONTAINER="$(singularity exec ${CONTAINER_PATH} /bin/bash -c 'ompi_info' | grep Prefix | awk '{ print $2 }')"
  OMPI_HOST="$(ompi_info | grep Prefix | awk '{ print $2 }')"

  # echo "Test container setup on mpibench using SRUN"
  # >&2 echo "Test container setup on mpibench using SRUN"
  # srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
  #   singularity exec --nv \
  #   $CONTAINER_PATH ./mpibench

  # # NOTE: this is causing PMIx errors: "PMIX ERROR: ERROR in file gds_ds12_lock_pthread.c at line 168"
  # >&2 echo "Test container setup on mpibench mounting host libs inside the container using SRUN"
  # echo "Test container setup on mpibench mounting host libs inside the container using SRUN"
  # srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
  #   singularity exec --nv \
  #   --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  #   $CONTAINER_PATH ./mpibench

  # # NOTE: this may cause PMIx problems as above
  # echo "Run itwinai's pytest with SRUN"
  # >&2 echo "Run itwinai's pytest with SRUN"
  # # pmix_v3: https://doc.vega.izum.si/mpi/#multi-node-jobs
  # srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
  #   singularity exec --nv \
  #   --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  #   $CONTAINER_PATH /bin/bash -c \
  #   'if [ $SLURM_PROCID  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'


  ######
  # mpirun container

  # >&2 echo "Test container setup on mpibench using MPIRUN"
  #  echo "Test container setup on mpibench using MPIRUN"
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core \
  #   singularity exec --nv \
  #   $CONTAINER_PATH ./mpibench

  # >&2 echo "Test container setup on mpibench mounting host libs inside the container using MPIRUN"
  #  echo "Test container setup on mpibench mounting host libs inside the container using MPIRUN"
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core \
  #   singularity exec --nv \
  #   --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  #   $CONTAINER_PATH ./mpibench

  echo
  echo "PATH: $PATH"
  echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
  echo

  # # This is working for deepspeed in a container with mpirun (single node)
  # echo "Run itwinai's pytest with MPIRUN (no mount)"
  # >&2 echo "Run itwinai's pytest with MPIRUN"
  
  # unset PYTHONPATH # Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages
  
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core  \
  #   -x HOROVOD_LOG_LEVEL=DEBUG -x HOROVOD_CYCLE_TIME=0.1  \
  #   singularity exec --nv  \
  #   $CONTAINER_PATH /bin/bash -c \
  #   'echo "Rank: $OMPI_COMM_WORLD_RANK, lrank: $OMPI_COMM_WORLD_LOCAL_RANK, Size: $OMPI_COMM_WORLD_SIZE" &&  \
  #   if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'


  # # This is working for deepspeed in a container with mpirun (single and multi node)
  # echo "Run itwinai's pytest with MPIRUN (no mount)"
  # >&2 echo "Run itwinai's pytest with MPIRUN"
  
  # unset PYTHONPATH # Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages
  #   #  --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core  \
  #   -x HOROVOD_LOG_LEVEL=DEBUG -x HOROVOD_CYCLE_TIME=0.1  \
  #   singularity exec --nv  \
  #   $CONTAINER_PATH /bin/bash -c \
  #   'echo "Rank: $OMPI_COMM_WORLD_RANK, lrank: $OMPI_COMM_WORLD_LOCAL_RANK, Size: $OMPI_COMM_WORLD_SIZE" &&  \
  #   if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'


  # ##########################################################
  # # This is working for deepspeed in a container with mpirun (single and multi node)
  # # This also works for horovod on single and multi node
  # echo "Run itwinai's pytest with MPIRUN (no mount)"
  # >&2 echo "Run itwinai's pytest with MPIRUN"
  
  # unset PYTHONPATH # Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages inside the container
  #   #  --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core  \
  #   -x HOROVOD_LOG_LEVEL=DEBUG -x HOROVOD_CYCLE_TIME=0.1  \
  #   singularity exec --nv  \
  #   $CONTAINER_PATH /bin/bash -c \
  #   'echo "Rank: $OMPI_COMM_WORLD_RANK, lrank: $OMPI_COMM_WORLD_LOCAL_RANK, Size: $OMPI_COMM_WORLD_SIZE" &&  \
  #   if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'
  # #####################################################################

  echo "Run itwinai's pytest with SRUN (no mount)"
  >&2 echo "Run itwinai's pytest with SRUN"
  unset PYTHONPATH # Avoid propagating PYTHONPATH to the singularity container, as it breaks the import of packages inside the container
  srun --mpi=pmix_v3 --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU --ntasks=$(($SLURM_GPUS_PER_NODE * $SLURM_NNODES)) \
    singularity exec --nv \
    $CONTAINER_PATH /bin/bash -c \
    'if [ $SLURM_PROCID  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'

  # # This hangs forever on gather ops
  # echo "Run itwinai's pytest with MPIRUN"
  # >&2 echo "Run itwinai's pytest with MPIRUN"
  # mpirun -H $HOSTFILE -np $TOTAL_PROCESSES --oversubscribe -mca pml ucx -mca btl ^uct,tcp,openib,vader --bind-to core  \
  #   singularity exec --nv \
  #   --bind "${OMPI_HOST}":"${OMPI_CONTAINER}" \
  #   $CONTAINER_PATH /bin/bash -c \
  #   'if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m $TRAINING_CMD'

  
  
  ################################ LOGIN NODE ################################
  exit

  mpirun -np 1 singularity exec --nv --hostname $(hostname) \
    $CONTAINER_PATH /bin/bash -c \
    'echo "Rank: $OMPI_COMM_WORLD_RANK, lrank: $OMPI_COMM_WORLD_LOCAL_RANK, Size: $OMPI_COMM_WORLD_SIZE" && unset PYTHONPATH && \
    if [ $OMPI_COMM_WORLD_RANK  -ne 0 ]; then exec > /dev/null 2>&1; fi; exec python -m pytest -m mpirun'
  

  # # Other
  # srun --cpu-bind=none --ntasks-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_GPU \
  #   singularity run --nv $CONTAINER_PATH \
  #   /bin/bash -c "$TRAINING_CMD"

else
  >&2 echo "ERROR: unrecognized \$DIST_MODE env variable"
  exit 1
fi
