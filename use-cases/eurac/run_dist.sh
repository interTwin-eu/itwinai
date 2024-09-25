#!/bin/bash

NUM_NODES=2
STRATEGY=horovod

srun --jobid 70634 --nodes=$NUM_NODES --ntasks-per-node=1 --gpus-per-node=2 \
bash -c "torchrun --nnodes=\$SLURM_NNODES \
--nproc-per-node=\$SLURM_GPUS_PER_NODE \
--rdzv_id=\$SLURM_JOB_ID \
--rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
--rdzv_backend=c10d \
--rdzv_endpoint=\$(scontrol show hostnames '$SLURM_JOB_NODELIST' | head -n 1):29500 \
\\$(which itwinai) exec-pipeline \
--config config.yaml \
--pipe-key training_pipeline \
-o training_pipeline.init_args.steps.2.init_args.strategy=$STRATEGY"
