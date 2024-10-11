# Noise Simulation for Gravitational Waves Detector (Virgo)

This repository contains code for simulating noise in the Virgo gravitational wave detector. The code is adapted from
[this notebook](https://github.com/interTwin-eu/DT-Virgo-notebooks/blob/main/WP_4_4/interTwin_wp_4.4_synthetic_data.ipynb)
available on the Virgo use case's [repository](https://github.com/interTwin-eu/DT-Virgo-notebooks).

## Installation

If running the pipeline directly on a node (or from your terminal),
first install the required libraries in the pre-existing itwinai environment using the following command:

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

Alternatively, you can run distributed training with SLURM using the dedicated `slurm.sh` job script:

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

When using MLFLow logger, you can visualize the logs in from the MLFlow UI:

```bash
mlflow ui --backend-store-uri mllogs/mlflow

# In background 
mlflow ui --backend-store-uri mllogs/mlflow > /dev/null 2>&1 &
```

## Running scaling tests

Scaling tests have been integrated into the eurac usecase to provide timing of experiments run and thus show the power of distributed model training and itwinai.
Refer to the files `runall.sh and scaling-test.sh.

Launch the scaling test:

```bash
bash scaling-test.sh
```

Generate plots for the outputs of the scaling tests
Once all jobs have completed, you can automatically generate scalability report
using itwinai's CLI:

```bash
# First, activate your Python virtual environment

# For more info run
itwinai scalability-report --help

# Generate a scalability report
itwinai scalability-report --pattern="^epoch.+\.csv$" \
    --plot-title "Virgo usecase scaling" --archive virgo_scaling
```

## Running HPO for Virgo Non-distributed

Hyperparameter optimization (HPO) is integrated into the pipeline using Ray Tune.
This allows you to run multiple trials and fine-tune model parameters efficiently.
HPO is configured to run multiple trials in parallel, but run those trials each in a non-distributed way.

To launch an HPO experiment, run

```bash
sbatch slurm_ray.sh
```

Make sure to adjust the #SBATCH directives in the script to specify the number of nodes, CPUs, and GPUs you want to allocate for the job.
The slurm_ray.sh script sets up a Ray cluster and runs hpo.py for hyperparameter tuning.This script creates a ray cluster,
and runs the python file `hpo.py`. You may change CLI variables for the python command to change parameters, such as the number of trials you want to run, or change the stopping criteria for the trials.
By default, trials monitor validation loss, and results are plotted once all trials are completed.
