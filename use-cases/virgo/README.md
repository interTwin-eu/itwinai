# Noise Simulation for Gravitational Waves Detector (Virgo)

## Installation

Before continuing, install the required libraries in the pre-existing itwinai environment.

```bash
pip install -r requirements.txt
```

## Training

You can run the whole pipeline in one shot, including dataset generation, or you can
execute it from the second step (after the synthetic dataset have been generated).

```bash
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline

# Run from the second step (use python-like slicing syntax).
# In this case, the dataset is loaded from "data/Image_dataset_synthetic_64x64.pkl"
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1:
```

Launch distributed training with SLURM using the dedicated `slurm.sh` job script:

```bash
# Distributed training with torch DistributedDataParallel
PYTHON_VENV="../../envAI_hdfml"
DIST_MODE="ddp"
RUN_NAME="ddp-virgo"
TRAINING_CMD="$PYTHON_VENV/bin/itwinai exec-pipeline --config config.yaml --steps 1: --pipe-key training_pipeline -o strategy=ddp"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",TRAINING_CMD="$TRAINING_CMD",PYTHON_VENV="$PYTHON_VENV" \
    slurm.sh
```

...and check the results in `job.out` and `job.err` log files.

To understand how to use all the distributed strategies supported by `itwinai`,
check the content of `runall.sh`:

```bash
bash runall.sh
```
