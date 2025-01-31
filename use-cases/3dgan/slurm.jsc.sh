#!/bin/bash

# SLURM jobscript for JSC systems

# general configuration of the job
#SBATCH --job-name=3dganTraining
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=24:00:00

# configure node and process count on the CM
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# load modules
ml --force purge
ml Stages/2024 GCC CUDA/12 cuDNN Python 
# ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
# ml Python CMake HDF5 PnetCDF libaio mpi4py

# shellcheck source=/dev/null
source ~/.bashrc

# Activate the environment
source ../../envAI_juwels/bin/activate

GAN_DATASET="/p/scratch/intertwin/datasets/cern" # 2_500_fullEnergy "/p/scratch/intertwin/datasets/cern/" exp_data
CHECKPOINT_DIR="checkpoints"
NUM_EPOCHS_PER_JOB=205

# # Function to submit the next job with dependency
# submit_next_job() {
#     sbatch --dependency=afterok:$1 $0
# }

# launch training
TRAINING_CMD="$(which itwinai) exec-pipeline --config config.yaml --pipe-key training_pipeline \
                -o max_epochs=$NUM_EPOCHS_PER_JOB \
                -o checkpoint_dir=$CHECKPOINT_DIR \
                -o num_nodes=$SLURM_NNODES \
                -o dataset_location=$GAN_DATASET"

# # launch inference
# INFERENCE_CMD="$(which itwinai) exec-pipeline --config config.yaml --pipe-key inference_pipeline \
#                 -o num_nodes=$SLURM_NNODES \
#                 -o dataset_location=$GAN_DATASET \
#                 -o model_checkpoint=$CHECKPOINT"

srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $TRAINING_CMD"

# # If training completes, submit the next job
# if [ $? -eq 0 ]; then
#     submit_next_job $SLURM_JOB_ID
# fi