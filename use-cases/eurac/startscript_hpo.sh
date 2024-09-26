#!/bin/bash

# SLURM settings for HPO with Ray on a cluster
#SBATCH --job-name=eurac-hpo
#SBATCH --account=intertwin
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err

# Load necessary modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

PYTHON_VENV="../../envAI_hdfml"
# Activate virtual environment
source $PYTHON_VENV/bin/activate

# Set Ray environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
export RAY_RESOURCES='{"CustomResource1": 2}'

# Job details for debugging
echo "Job details:"
echo "Time: $(date)"
echo "Executing command: $COMMAND"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Set up Ray cluster
num_gpus=$SLURM_GPUS_PER_NODE
export TUNE_MAX_PENDING_TRIALS_PG=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export RAY_USAGE_STATS_DISABLE=1

# Get the head node IP and start Ray head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node=${nodes_array[0]}
port=29500
export ip_head="$head_node:$port"
ray start --head --node-ip-address=$head_node --port=$port --num-cpus=$SLURM_CPUS_ON_NODE --num-gpus=$num_gpus --block &

sleep 10  # Wait for the head node to initialize

# Start Ray worker nodes
for i in $(seq 1 $(($SLURM_NNODES - 1))); do
    node=${nodes_array[$i]}
    echo "Starting worker node $i at $node"
    srun --nodes=1 --ntasks=1 -w $node ray start --address=$ip_head --redis-password='5241590000000000' --num-cpus=$SLURM_CPUS_ON_NODE --num-gpus=$num_gpus --block &
    sleep 5
done

echo "Ray cluster setup complete"

# Command to run
python -u train_hpo.py --num-samples 10 --max-iterations 50 --ngpus 1 --scheduler ASHA

# Shutdown Ray after completion
ray stop
