#!/bin/bash

rm -rf slurm_logs mllogs
mkdir slurm_logs

# ========== FOR JSC ==========
SLURM_SCRIPT="slurm.jsc.sh"

# ========== FOR Vega ==========
# SLURM_SCRIPT="slurm.vega.sh"

# Launch experiments

# 1 worker
sbatch --wait --nodes=1 --output=slurm_logs/1_worker.out --error=slurm_logs/1_worker.err \
    --gres=gpu:1 --gpus-per-node=1 --time=02:00:00 $SLURM_SCRIPT

# 4, 8, 16... workers
sbatch --wait --nodes=1 --output=slurm_logs/4_worker.out --error=slurm_logs/4_worker.err --time=00:59:00 $SLURM_SCRIPT
sbatch --wait --nodes=2 --output=slurm_logs/8_worker.out --error=slurm_logs/8_worker.err $SLURM_SCRIPT
sbatch --wait --nodes=4 --output=slurm_logs/16_worker.out --error=slurm_logs/16_worker.err $SLURM_SCRIPT
sbatch --wait --nodes=8 --output=slurm_logs/32_worker.out --error=slurm_logs/32_worker.err $SLURM_SCRIPT