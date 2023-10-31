#!/bin/bash

# general configuration of the job
#SBATCH --job-name=itwinaiTraining
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4

#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# parameters
debug=false # display debug info

CONTAINERPATH="/p/project/intertwin/zoechbauer1/T6.5-AI-and-ML/containers/apptainer/itwinai.sif"

SLURM_EXECUTION=false

#switch to use case folder
cd use-cases/mnist/torch


### DO NOT MODIFY THE SCRIPT BELOW THIS LINE ###

# set modules and envs
ml GCC/11.3.0 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Apptainer-Tools/2023
# source $SLURM_SUBMIT_DIR/torch_env/bin/activate

# set env vars
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# job info 
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE" 
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi



#This is to overwrite the default run command in the container, e.g.:

#EXEC="python train.py -p pipeline.yaml --download-only" #for bash
# if [ "$SLURM_EXECUTION" = true ]; then
#     srun --cpu-bind=none bash -c "apptainer exec --nv \
#         $CONTAINERPATH \
#         $EXEC"
# else
#     apptainer exec --nv \
#         $CONTAINERPATH \
#         $EXEC
# fi

#Choose SLURM execution or bash script execution
if [ "$SLURM_EXECUTION" = true ]; then
    srun --cpu-bind=none bash -c "apptainer run --nv \
        $CONTAINERPATH"

else
    apptainer run --nv \
        $CONTAINERPATH
fi

#eof



