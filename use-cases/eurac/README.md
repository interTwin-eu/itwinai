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

## Running scaling tests
Scaling tests have been integrated into the eurac usecase to provide timing of experiments run and ths show the power of distributed model training and itwinai. Refer to the following files `runall.sh , scaling-test.sh, torch_dist_final_scaling.py`.

Launch the scaling test:

```bash
bash scaling-test.sh
```

Generate plots for the outputs of the scaling tests
Once all jobs have completed, you can automatically generate scalability report
using itwinai's CLI:

```bash
# First, activate you Python virtual environment

# For more info run
itwinai scalability-report --help

# Generate a scalability report
itwinai scalability-report --pattern="^epoch.+\.csv$" \
    --plot-title "Eurac usecase scaling" --archive eurac_scaling
```


## Running HPO for EURAC Non-distributed

HPO has been implemented using Ray tuner to run in a non distributed environment. Refer to `train_hpo.py` file which was adapted from `train.py`. The current HPO parameters include learning rate(lr) and batch_size. 

Launch the hpo expirement:

```bash
sbatch startscript_hpo.sh
```

Visualize the HPO results by running `python visualize_hpo.py`. Adjust the `main_dir = '/p/home/jusers/<username>/hdfml/ray_results/<specific run folder name>'` accordingly based on the run folder name, the results path can be got from your slurm output file at the top.