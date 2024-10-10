# EURAC use case

## Installation

First, make sure to install itwinai from this branch!
Use the [developer installation instructions](https://github.com/interTwin-eu/itwinai/#installation-for-developers).

Then install the dependencies specific to this use case by first entering the 
folder and then installing the dependencies with pip:

```bash
cd use-cases/eurac
pip install -r requirements.txt
```

## Interactive session on SLURM

Allocate 4 GPUs on a compute node and run distributed algorithms:
see "[Distributed training on a single node (interactive)](https://github.com/interTwin-eu/itwinai/tree/main/tutorials/distributed-ml/torch-tutorial-0-basics#distributed-training-on-a-single-node-interactive)."

```bash

salloc --partition=batch --nodes=1 --account=intertwin  --gres=gpu:4 --time=1:59:00

srun --jobid XXXX --overlap --pty /bin/bash

ml --force purge

ml Stages/2024 GCC OpenMPI CUDA/12 MPI-settings/CUDA Python HDF5 PnetCDF libaio mpi4py

source ../../../hython-dev/bin/activate

torchrun --standalone --nnodes=1 --nproc-per-node=gpu dist-train.py

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

Launch distributed training with SLURM using the dedicated `runall.sh` job script:

Train LSTM

```bash
# Distributed training with torch DistributedDataParallel
./runall.sh config.yaml
```

Train ConvLSTM

```bash
# Distributed training with torch DistributedDataParallel
./run.sh config_conv.yaml
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

## Exporting a local mlflow run to the on EGI cloud mlflow remote tracking server

Install [mlflow-export-import](https://github.com/mlflow/mlflow-export-import)

```bash
export MLFLOW_TRACKING_INSECURE_TLS='true'
export MLFLOW_TRACKING_USERNAME='iacopo.ferrario@eurac.edu'
export MLFLOW_TRACKING_PASSWORD='YOUR_PWD'
export MLFLOW_TRACKING_URI='https://mlflow.intertwin.fedcloud.eu/'
```

Assuming the working directory is the eurac usecase, export the run-id from the local mlflow logs directory. This will also export all the associated artifacts (included models and model weights)


```bash
copy-run --run-id 27a81c42c2cb40dfb7505032f1ac1ef5 --experiment-name "drought use case lstm" --src-mlflow-uri mllogs/mlflow --dst-mlflow-uri https://mlflow.intertwin.fedcloud.eu/
```

## Loading a pre-trained model from the mlflow registry on the local host for prediction/fine-tuning

```bash
export MLFLOW_TRACKING_INSECURE_TLS='true'
export MLFLOW_TRACKING_USERNAME='iacopo.ferrario@eurac.edu'
export MLFLOW_TRACKING_PASSWORD='YOUR_PWD'
export MLFLOW_TRACKING_URI='https://mlflow.intertwin.fedcloud.eu/'
```

```python

import mlflow

logged_model = 'runs:/1811bd3835d54585b6376dd97f6687a5/LSTM'

loaded_model = mlflow.pyfunc.load_model(logged_model)

```

> :warning: While the model is loading an error occurs **RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.**
> Possible reasons due to package version mismatch https://github.com/mlflow/mlflow/issues/4903.
