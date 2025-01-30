srun --cpu-bind=none --ntasks-per-node=1 \
    bash -c "torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=151152 \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    python train.py"
