# EURAC use case

## Installation

First, make sure to install itwinai from this branch!
Use the [developer installation instructions](https://github.com/interTwin-eu/itwinai/tree/usecase_eurac?tab=readme-ov-file#installation-for-developers).

Then:

```bash
pip install -r requirements.txt
```

## Interactive session on SLURM

Allocate 4 GPUs on a compute node and run distributed algorithms:
see "[Distributed training on a single node (interactive)](https://github.com/interTwin-eu/itwinai/tree/main/tutorials/distributed-ml/torch-tutorial-0-basics#distributed-training-on-a-single-node-interactive)."


## Training

You can run the whole pipeline in one shot, including dataset generation, or you can
execute it from the second step (after the synthetic dataset have been generated).

```bash
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline

# Run from the second step (use python-like slicing syntax).
# In this case, the dataset is loaded from "data/Image_dataset_synthetic_64x64.pkl"
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1:
```

Launch distributed training with SLURM using the dedicated `run.sh` job script:

```bash
# Distributed training with torch DistributedDataParallel
./run.sh
```