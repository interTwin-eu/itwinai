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

You can run the training in a distributed manner using all strategies by running `runall.sh`.
This will launch jobs for all the strategies and log their outputs into the logs_slurm folder.

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

Scaling tests provide information about how well the different distributed strategies scale.
We have integrated them into this use case and you can run them using the scaling-test.sh script.´:

```bash
bash scaling-test.sh
```

To generate the plots, refer to the [Scaling-Test Tutorial](https://github.com/interTwin-eu/itwinai/tree/main/tutorials/distributed-ml/torch-scaling-test#analyze-results).

## Running HPO for Virgo Non-distributed

Hyperparameter optimization (HPO) is integrated into the pipeline using Ray Tune.
This allows you to run multiple trials and fine-tune model parameters efficiently.
HPO is configured to run multiple trials in parallel, but run those trials each in a non-distributed way.

To launch an HPO experiment, run

```bash
sbatch slurm_ray.sh
```

This script sets up a Ray cluster and runs `hpo.py` for hyperparameter tuning.
You may change CLI variables for `hpo.py` to change parameters,
such as the number of trials you want to run, to change the stopping criteria for the trials or to set a
different metric on which ray will evaluate trial results.
By default, trials monitor validation loss, and results are plotted once all trials are completed.

## Generating Synthetic Data for the Virgo Use Case

This project includes another SLURM job script, `synthetic_data_gen/data_generation.sh`, that allows
users to generate synthetic dataset for the Virgo gravitational wave detector use case.
This step is typically not required unless you need to create new synthetic datasets.

The synthetic data is generated using a Python script, `file_gen.py`, which creates multiple files
containing simulated data. Each file is a pickled pandas dataframe containing `datapoints_per_file`
datapoints (defaults to 500), each
one representing a set of time series for main and strain detector channels. 

If you need to generate a new dataset, you can run the SLURM script with the following command:

```bash
sbatch data_generation.sh
```

The script will generate multiple data files and store them in separate folders, which are
created in the `target_folder_name` directory.

The generated pickle files are organized in a set of nested folders to avoid creating too many
files in the same folder. To generate such folders and its files we use SLURM 
[job arrays](https://slurm.schedmd.com/job_array.html).
Each SLURM array job will create its own folder and populate it with the synthetic data files.
The number of files created in each folder can be customized by setting the `NUM_FILES` environment
variablebefore submitting the job.
For example, to generate 50 files per array job, you can run:

```bash
export NUM_FILES=50
sbatch data_generation.sh
```

If you do not specify `NUM_FILES`, the script will default to creating 100 files per folder.
