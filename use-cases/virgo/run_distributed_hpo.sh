#!/bin/bash

# Job configuration
#SBATCH --job-name=ray_tune_hpo    
#SBATCH --account=intertwin       
#SBATCH --time 0:30:00

# Resources allocation
#SBATCH --cpus-per-task=16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --gres=gpu:4

# Output and error logs
#SBATCH -o logs_slurm/job.out
#SBATCH -e logs_slurm/job.err

# Load environment modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

# Set and activate virtual environment
PYTHON_VENV="../../envAI_hdfml"
source $PYTHON_VENV/bin/activate

#pip install hpbandster ConfigSpace

# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}  # TODO: test if this makes a difference

num_gpus=$SLURM_GPUS_PER_NODE

## Limit number of max pending trials
#export TUNE_MAX_PENDING_TRIALS_PG=$(($SLURM_NNODES * 4))

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1


#########   Set up Ray cluster   ########

# Get the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Set head node at port
head_node=${nodes_array[0]}
port=7639

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &

sleep 10  # Wait for head node to intitialize


# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --redis-password='5241580000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 5
done
echo All Ray workers started.

##############################################################################################


# Run the Python script using Ray
echo 'Starting HPO.'

#python -u train_hpo.py --num-samples 10 --max-iterations 50
python train_hpo-no-hpo.py

# Shutdown Ray after completion
ray stop
