#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=intertwin
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00

# configure node and process count on the CM
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# parallelization STRATEGY (ddp, horovod, deepspeed)
STRATEGY='ddp'

# parameters
debug=false # do debug
bs=32       # batch-size
epochs=10    # epochs
lr=0.01     # learning rate


# set modules
ml --force purge

ml Stages/2022 NVHPC/22.1 ParaStationMPI/5.5.0-1-mt NCCL/2.11.4-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5
ml Python/3.9.6 CMake HDF5 PnetCDF libaio/0.3.112 mpi-settings/CUDA

# set env
source /p/project/intertwin/rakesh/T6.5-AI-and-ML/dist_trainer/envAI_hdfml/bin/activate

# sleep a sec
sleep 1

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
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

#launch
# srun python train.py --STRATEGY hvd --n_workers_per_node $SLURM_GPUS_PER_NODE

if [[ $STRATEGY == *"horovod"* ]];
then
  echo "NOT IMPLEMENTED"
  #   COMMAND="horovod_trainer.py"
    
  #   EXEC="$COMMAND \
  #  --batch-size $bs \
  #  --epochs $epochs \
  #  --lr $lr \
  #  --data-dir $dataDir"
   
  #  # MB: how local worker processes are spawned?
  #  srun --cpu-bind=none python3 -u $EXEC

elif [[ $STRATEGY ==  *"ddp"* ]];
then
    COMMAND="ex0_multiple_torch_strategies.py --strategy ddp"
 
    EXEC="$COMMAND"
  #  --batch-size $bs \
  #  --epochs $epochs \
  #  --lr $lr \
  #  --nworker $SLURM_CPUS_PER_TASK \
  #  --data-dir $dataDir"
 
   srun --cpu-bind=none bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $EXEC"

else
  echo "NOT IMPLEMENTED"
  #  COMMAND="DS_trainer.py"

  #  EXEC=$COMMAND" --batch-size $bs
  # --epochs $epochs
  # --nworker $SLURM_CPUS_PER_TASK
  # --data-dir $dataDir"

  #  #### do not change this part
  #  # create node-list
  #  sysN=$(eval "scontrol show hostnames")
  #  for i in $sysN; do
  #    x+=\"$i\":[$CUDA_VISIBLE_DEVICES],
  #  done
  #  WID=`echo {${x::-1}} | base64 -w 0`

  #  # modify config file with parameters
  #  sed -i "2s|.*|  \"train_micro_batch_size_per_gpu\": ${bs},|" DS_config.json
  #  sed -i "7s|.*|      \"lr\": ${lr}|" DS_config.json
  #  ####

  #  # launch
  #  srun python -m deepspeed.launcher.launch \
  #    --node_rank $SLURM_PROCID \
  #    --master_addr ${SLURMD_NODENAME}i \
  #    --master_port 29500 \
  #    --world_info $WID \
  #    $EXEC --deepspeed_mpi --deepspeed_config DS_config.json
  
fi
