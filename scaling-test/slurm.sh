#!/bin/bash

# SLURM jobscript for JSC systems

# Job configuration
#SBATCH --job-name=comm_analysis_horovod
#SBATCH --account=intertwin
#SBATCH --output=logs_slurm/horovod.out
#SBATCH --error=logs_slurm/horovod.err
#SBATCH --time=00:10:00

# Resources allocation
#SBATCH --partition=batch
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --exclusive

# Load environment modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

PYTHON_VENV="../envAI_hdfml"
source $PYTHON_VENV/bin/activate

# srun --cpu-bind=none --ntasks-per-node=1 \
# 	bash -c "torchrun \
# 	--log_dir='logs_torchrun' \
# 	--nnodes=$SLURM_NNODES \
# 	--nproc_per_node=$SLURM_GPUS_PER_NODE \
# 	--rdzv_id=$SLURM_JOB_ID \
# 	--rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
# 	--rdzv_backend=c10d \
# 	--rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)':29500 \
# 	profiler.py --epochs 5 --strategy deepspeed"


export CUDA_VISIBLE_DEVICES="0,1,2,3"
srun --cpu-bind=none --nodes=$SLURM_NNODES \
	--ntasks-per-node=$SLURM_GPUS_PER_NODE \
	--cpus-per-task=$SLURM_CPUS_PER_GPU \
	python profiler.py --epochs 5 --strategy horovod
