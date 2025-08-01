ml --force purge
ml Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py git

OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" -gt 0 ] ; then
  OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
  || [ "$SYSTEMNAME" = juwels ] \
  || [ "$SYSTEMNAME" = jurecadc ] \
  || [ "$SYSTEMNAME" = jusuf ]; then
# Allow communication over InfiniBand cells on JSC machines.
MASTER_ADDR="$MASTER_ADDR"i
fi
MASTER_PORT=54123

# Prevent NCCL not figuring out how to initialize.
NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
GLOO_SOCKET_IFNAME=ib0

launch_torchrun() {
  srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c 'torchrun_jsc \
    --log_dir="logs_torchrun" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_conf=is_host=$(( SLURM_NODEID == 0 ? 1 : 0 )) \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
    "$1"
  ' bash "$1"
  }

# Exporting at the end to comply with SC2155
export OMP_NUM_THREADS MASTER_ADDR MASTER_PORT NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
