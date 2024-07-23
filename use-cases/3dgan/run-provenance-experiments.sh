#!/bin/bash

rm -rf slurm_logs mllogs
mkdir slurm_logs

source ../../.venv-pytorch/bin/activate
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline 1> slurm_logs/1_worker.out 2> slurm_logs/1_worker.err

sbatch --wait --nodes=1 --output=slurm_logs/4_worker.out --error=slurm_logs/4_worker.err slurm.vega.sh
sbatch --wait --nodes=2 --output=slurm_logs/8_worker.out --error=slurm_logs/8_worker.err slurm.vega.sh
# sbatch --wait --nodes=4 --output=slurm_logs/16_worker.out --error=slurm_logs/16_worker.err slurm.vega.sh
# sbatch --wait --nodes=8 --output=slurm_logs/32_worker.out --error=slurm_logs/32_worker.err slurm.vega.sh