ml --force purge
ml CMake/3.29.3-GCCcore-13.3.0
ml mpi4py/3.1.5
ml OpenMPI/4.1.6-GCC-13.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
ml CUDA/12.6.0
ml NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
ml Python/3.12.3-GCCcore-13.3.0

export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
 
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=54123
