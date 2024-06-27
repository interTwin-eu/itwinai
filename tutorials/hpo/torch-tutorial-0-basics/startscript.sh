#!/bin/bash

# general configuration of the job
#SBATCH --job-name=HPO-ASHA
#SBATCH --account=intertwin
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

# Load environment modules
ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

# command
dataDir=/p/project1/intertwin/ruettgers1/HPO/itwinai/tutorials/hpo/torch-tutorial-0-basics/datasets/
COMMAND="hpo.py --num-samples 12 --ngpus 2 --max-iterations 2 --scheduler ASHA"
EXEC="$COMMAND \
  --nworker $SLURM_CPUS_PER_TASK \
  --data-dir $dataDir"

# set modules
unknown HPC detected!
ml

# set env
source /p/project1/intertwin/ruettgers1/HPO/itwinai/envAI_hdfml/bin/activate

# set vars
export NCCL_DEBUG=INFO
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# job info 
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" 
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"

# extension
num_gpus=$SLURM_GPUS_PER_NODE

export TUNE_MAX_PENDING_TRIALS_PG=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export RAY_USAGE_STATS_DISABLE=1

set -x

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
port=29500

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --redis-password='5241590000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 5
done

echo "Ready"

# launch
python -u $EXEC

#eof
