# xtclim
## ML-based extreme events detection and characterization (CERFACS)
**Integration author(s)**: Rakesh Sarma (Juelich), Jarl Sondre Sæther (CERN), Matteo Bunino (CERN)

The code is adapted from CERFACS' [repository](https://github.com/cerfacs-globc/xtclim/tree/master).
The implementation of a pipeline with itwinai framework is shown below. 

## Method 
Convolutional Variational AutoEncoder.

## Input
"3D daily images", daily screenshots of Europe for three climate variables (maximum temperature, precipitation, wind).

## Output 
Error between original and reconstructed image: postprocessed for analysis in the `scenario_season_comparison.ipynb` file.

## Idea 
The more unusual an image (anomaly), the higher error.

## Information on files

In the preprocessing folder, the `preprocess_functions_2d_ssp.py` class loads NetCDF files from a `data` folder, which has to be specified in `dataset_root` in the config file `pipeline.yaml` (please change the location). The data can be found [here](https://b2drop.eudat.eu/s/rtAadDNYDWBkxjJ). The given class normalizes and adjusts the data for the network. The function `preprocess_2d_seasons.py` splits the data into seasonal files. Preprocessed data is stored in the `input` folder.

The file `train.py` trains the network. Caution: It will overwrite the weights of the network already saved in outputs (unless you change the path name `outputs/cvae_model_3d.pth` in the script).

The `anomaly.py` file evaluates the network on the available datasets - train, test, and projection.

## Installation

Please follow the documentation to install the itwinai environment.
After that, install the required libraries within the itwinai environment with:

```bash
pip install -r requirements.txt
```

## How to launch pipeline locally

The config file `pipeline.yaml` contains all the steps to execute the workflow. 
This file also contains all the seasons, and a separate run is launched for each season.
You can launch the pipeline through `train.py` from the root of the repository with:

```bash
python train.py

```

## How to launch pipeline on an HPC system

The `startscript` job script can be used to launch a pipeline with SLURM on an HPC system.
These steps should be followed to export the environment variables required by the script.

```bash
# Distributed training with torch DistributedDataParallel
PYTHON_VENV=".venv"
DIST_MODE="ddp"
RUN_NAME="ddp-cerfacs"
sbatch --export=ALL,DIST_MODE="$DIST_MODE",RUN_NAME="$RUN_NAME",PYTHON_VENV="$PYTHON_VENV" \
    startscript
```

The results and/or errors are available in `job.out` and `job.err` log files.
Training and inference steps are defined in the pipeline, where distributed resources
are exploited in both the steps.

With MLFLow logger, the logs can be visualized in the MLFlow UI:

```bash
itwinai mlflow-ui --path mllogs/mlflow --port 5000 --host 127.0.0.1

# In background
itwinai mlflow-ui --path mllogs/mlflow --port 5000 --host 127.0.0.1 &
```

### Hyperparameter Optimization (HPO)

The repository also provides functionality to perform HPO with Ray. With HPO, 
multiple trials with different hyperparameter configurations are run in a distributed 
infrastructure, typically in an HPC environment. This allows searching for optimal 
configurations that provide the minimal/maximal loss for the investigated network.
The `hpo.py` file contains the implementation, which launches the `pipeline.yaml` pipeline.
To launch an HPO experiment, simply run:
```bash
sbatch slurm_ray.sh
```
The parsing arguments to the `hpo.py` file can be changed to customize the required parameters
that need to be considered in the HPO process. 
