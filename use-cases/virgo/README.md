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

## Training Pipelines

This repository offers two main approaches for training based on the dataset size:

### Small Dataset Pipeline

This pipeline allows you to generate a small synthetic dataset on-the-fly as part of the training process.
It's suited for quick tests and debugging where the entire workflow stays self-contained in memory.
The dataset generation step can also be skipped for subsequent runs.

To run the entire pipeline, including dataset generation, use the following command:

```bash
itwinai exec-pipeline --config small_ds_config.yaml --pipe-key training_pipeline
```

If you’ve already generated the dataset in a previous run, you can skip the dataset generation step by executing the following command:

```bash
itwinai exec-pipeline --config small_ds_config.yaml --pipe-key training_pipeline --steps 1:
```

This will load the dataset from memory and proceed with the training steps.

### Large Dataset Pipeline

The large dataset pipeline is designed to handle massive datasets that are stored on disk. To generate this data, this project includes another SLURM job script,
`synthetic_data_gen/data_generation.sh`, which generates a synthetic dataset for the Virgo gravitational wave detector use case.

The synthetic data is generated using a Python script, `file_gen.py`, which creates multiple files
containing simulated data. Each file is a pickled pandas dataframe containing `datapoints_per_file`
datapoints (defaults to 500), each
one representing a set of time series for main and strain detector channels.

If you need to generate a new dataset, you can run the SLURM script with the following command:

```bash
sbatch synthetic_data_gen/data_generation.sh
```

The script will generate multiple data files and store them in separate folders, which are
created in the `target_folder_name` directory.

The generated pickle files are organized in a set of nested folders to avoid creating too many
files in the same folder. To generate such folders and its files we use SLURM
[job arrays](https://slurm.schedmd.com/job_array.html).
Each SLURM array job will create its own folder and populate it with the synthetic data files.
The number of files created in each folder can be customized by setting the `NUM_FILES` environment
variable before submitting the job.
For example, to generate 50 files per array job, you can run:

```bash
export NUM_FILES=50
sbatch synthetic_data_gen/data_generation.sh
```

If you do not specify `NUM_FILES`, the script will default to creating 100 files per folder.

Once the dataset is generated, you can proceed with training:

```bash
itwinai exec-pipeline --config large_ds_config.yaml --pipe-key training_pipeline
```

You can also run the training in a distributed manner using all strategies by running runall.sh:

```bash
bash runall.sh
```

Change the `$TRAINING_CMD` variable in `runall.sh` to reflect the pipeline you wish to run, as explained above.
This will launch jobs for all the strategies and log their outputs into the logs_slurm folder.

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
