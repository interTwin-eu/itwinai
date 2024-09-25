#!/bin/bash

# Job configuration
#SBATCH --job-name=horovod
#SBATCH --account=intertwin
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=2      # Number of tasks per node (GPUs per node)
#SBATCH --gpus-per-node=2        # Number of GPUs per node
#SBATCH --cpus-per-gpu=2
#SBATCH --exclusive 

module load Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py
source ../../envAI_hdfml/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

# This needs to be modified depending on how many GPUs you have per node
export CUDA_VISIBLE_DEVICES=0,1

if [ -z "$EPOCHS" ]; then 
	EPOCHS=2
fi
echo "Starting slurm script with strategy: horovod and with $EPOCHS epochs"

# Since Horovod is MPI-based, it does not use 'torchrun' like the others
srun 	--cpu-bind=none \
	--ntasks-per-node=$SLURM_GPUS_PER_NODE \
	--cpus-per-task=$SLURM_CPUS_PER_GPU \
	$(which itwinai) exec-pipeline \
	--config config.yaml \
	--pipe-key training_pipeline \
	-o strategy=horovod
			# python -u torch_dist_final_scaling.py \
			# --epochs $EPOCHS \
			# --strategy horovod

