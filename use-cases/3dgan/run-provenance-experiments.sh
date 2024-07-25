#!/bin/bash

rm -rf slurm_logs mllogs
mkdir slurm_logs

# ========== FOR JSC ==========
# ml --force purge
# ml Stages/2024 GCC CUDA/12 cuDNN Python

# SLURM_SCRIPT="slurm.jsc.sh"
# ========== FOR JSC ==========

# ========== FOR Vega ==========
SLURM_SCRIPT="slurm.vega.sh"
# ========== FOR Vega ==========

source ../../envAI_hdfml/bin/activate
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline 1> slurm_logs/1_worker.out 2> slurm_logs/1_worker.err



sbatch --wait --nodes=1 --output=slurm_logs/4_worker.out --error=slurm_logs/4_worker.err $SLURM_SCRIPT
sbatch --wait --nodes=2 --output=slurm_logs/8_worker.out --error=slurm_logs/8_worker.err $SLURM_SCRIPT
sbatch --wait --nodes=4 --output=slurm_logs/16_worker.out --error=slurm_logs/16_worker.err $SLURM_SCRIPT
sbatch --wait --nodes=8 --output=slurm_logs/32_worker.out --error=slurm_logs/32_worker.err $SLURM_SCRIPT