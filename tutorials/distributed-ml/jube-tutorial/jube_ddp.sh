#!/bin/bash

# general configuration of the job
#SBATCH --job-name=JUBE_DDP
#SBATCH --account=#ACC#
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=#TIMELIM#

# configure node and process count on the CM
#SBATCH --partition=#QUEUE#
#SBATCH --nodes=#NODES#
#SBATCH --cpus-per-task=#NW#
#SBATCH --gpus-per-node=#NGPU#
#SBATCH --exclusive

# command
dataDir='/p/scratch/intertwin/datasets/imagenet/ILSVRC2012'
TRAINING_CMD="#SCRIPT#"
EXEC="$TRAINING_CMD \
  --nworker $SLURM_CPUS_PER_TASK \
  --data-dir $dataDir"

# set modules
ml GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# set env
source ../dist_trainer_v2/envAI_hdfml/bin/activate

# set vars
export NCCL_DEBUG=INFO
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# job info
echo "DEBUG: TIME: $(date)"
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

# launch
srun --cpu-bind=none --ntasks-per-node=1 bash -c "torchrun \
  --log_dir='logs' \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d  \
  --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
  --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
  $EXEC"

#eof
