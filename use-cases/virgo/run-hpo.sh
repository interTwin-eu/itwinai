#!/bin/bash

# Job configuration
#SBATCH --job-name=ray_tune_hpo    
#SBATCH --account=intertwin       
#SBATCH --time 0:30:00

# Resources allocation
#SBATCH --cpus-per-task=1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --exclusive

# Output and error logs
#SBATCH -o logs_slurm/hpo-job.out
#SBATCH -e logs_slurm/hpo-job.err

# Load environment modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

# Set and activate virtual environment
PYTHON_VENV="../../envAI_hdfml"
source $PYTHON_VENV/bin/activate

# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"

num_gpus=$SLURM_GPUS_PER_NODE

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1

# This tells Tune to not change the working directory to the trial directory
# which makes relative paths accessible from inside a trial
export RAY_CHDIR_TO_TRIAL_DIR=0

#########   Set up Ray cluster   ########

# Get the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Set head node at port
head_node=${nodes_array[0]}
port=7639

# This is so that the ray.init() command called from the hpo.py script knows which ports to connect to
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

python hpo.py --num_samples 3 --max_iterations 2

# Shutdown Ray after completion
ray stop
