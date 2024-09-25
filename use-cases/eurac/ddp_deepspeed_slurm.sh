#!/bin/bash

# SLURM job script

# Job configuration
#SBATCH --job-name=distributed_training
#SBATCH --account=intertwin
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus-per-node=2        # Number of GPUs per node
#SBATCH --cpus-per-gpu=2
#SBATCH --exclusive 

module load Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py
source ../../envAI_hdfml/bin/activate

# Debugging
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"
# export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

if [ -z "$STRATEGY" ]; then 
	STRATEGY=ddp
fi
if [ -z "$EPOCHS" ]; then 
	EPOCHS=2
fi
echo "Starting slurm script with strategy: $STRATEGY and with $EPOCHS epochs"

# Run the training command
srun --ntasks-per-node=1 --gpus-per-node=$SLURM_GPUS_PER_NODE \
	torchrun --nnodes=$SLURM_NNODES \
	--nproc-per-node=$SLURM_GPUS_PER_NODE \
	--rdzv_id=$SLURM_JOB_ID \
	--rdzv_conf=is_host=$(((SLURM_NODEID)) && echo 0 || echo 1) \
	--rdzv_backend=c10d \
	--rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):29500 \
	$(which itwinai) exec-pipeline \
	--config config.yaml \
	--pipe-key training_pipeline \
	-o strategy=$STRATEGY

	# torch_dist_final_scaling.py \
	# --epochs $EPOCHS \
	# --strategy $STRATEGY
