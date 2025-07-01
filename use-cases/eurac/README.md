# EURAC use case

**Integration authors**: Jarl Sondre Saether (CERN), Henry Mutegeki (CERN), Iacopo Ferrario (EURAC), Matteo Bunino (CERN)

## Installation

First, make sure to install itwinai from this branch!
Use the [developer installation instructions](https://github.com/interTwin-eu/itwinai/#installation-for-developers).

Then install the dependencies specific to this use case by first entering the
folder and then installing the dependencies with pip:

```bash
cd use-cases/eurac
pip install -r requirements.txt
```

## Training

You can run the RNN pipeline with the following command:

```bash
itwinai exec-pipeline +pipe_key=rnn_training_pipeline
```

If you want to use the Conv pipeline instead, you can replace `rnn_training_pipeline`
with `conv_training_pipeline`.

### Training using SLURM

If you wish to train the model using SLURM, you can use the `itwinai` SLURM script
builder with the following command to generate a preview of the script:

```bash
itwinai generate-slurm -c slurm_config.yaml --no-save-script --no-submit-job
```

If you are happy with the SLURM script, you can run it either by removing
`--no-submit-job` and let the builder submit it for you, or you can remove
`--no-save-script`—allowing the builder to store the script for you—and then running
the script yourself using `sbatch <path/to/script>`. 

### Scaling Tests and "runall"

Scaling tests provide information about how well the different distributed strategies
scale. We have integrated them into this use case and you can run them using the
`slurm.py` file. The format is very similar to the `itwinai generate-slurm` command,
and you can even pass it the configuration file, but it will overwrite some of the
parameters automatically—such as `std_out`, `err_out` and `job_name`. 

You can run all strategies by setting `--mode` to `runall` and you can run scaling
tests by setting `--mode` to `scaling-test` and specifying `scalability_nodes` in the
configuration.

## Running HPO for EURAC Non-distributed

Hyperparameter optimization (HPO) is integrated into the pipeline using Ray Tune.
This allows you to run multiple trials and fine-tune model parameters efficiently.
HPO is configured to run multiple trials in parallel, but run those trials each in a non-distributed way.

To launch an HPO experiment, run

```bash
sbatch slurm_ray.sh
```

This script sets up a Ray cluster and runs `hpo.py` for hyperparameter tuning.
You may change CLI variables for `hpo.py` to change parameters,
such as the number of trials you want to run, to change the stopping criteria for the trials or to set a different metric on which ray will evaluate trial results.
By default, trials monitor validation loss, and results are plotted once all trials are completed.

## Exporting a local MLFlow run to the EGI cloud MLFlow remote tracking server

Install [mlflow-export-import](https://github.com/mlflow/mlflow-export-import)

```bash
export MLFLOW_TRACKING_INSECURE_TLS='true'
export MLFLOW_TRACKING_USERNAME='iacopo.ferrario@eurac.edu'
export MLFLOW_TRACKING_PASSWORD='YOUR_PWD'
export MLFLOW_TRACKING_URI='https://mlflow.intertwin.fedcloud.eu/'
```

Assuming the working directory is the EURAC use case, export the run-id from the
local mlflow logs directory. This will also export all the associated artifacts
(including models and model weights)

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

> [!WARNING] While the model is loading an error occurs **RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.**
> Possible reasons due to package version mismatch <https://github.com/mlflow/mlflow/issues/4903>.
