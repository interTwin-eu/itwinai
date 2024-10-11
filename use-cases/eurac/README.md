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

## Training

You can run the RNN pipeline with the following command:

```bash
itwinai exec-pipeline --config config.yaml --pipe-key rnn_training_pipeline
```

If you want to use the Conv pipeline instead, you can replace `rnn_training_pipeline` 
with `conv_training_pipeline`.

### Distributed runs
You can run the training in a distributed manner using all strategies by running 
`runall.sh`. This will launch jobs for all the strategies and log their outputs into the 
`logs_slurm` folder. You can pass arguments by prepending environment variables. E.g. if 
you want to turn debug-mode on, you pass `DEBUG=false` as follows: 

```bash
DEBUG=false ./runall.sh
```
The same can be done with any other variables you might want to change. You can see 
all the variables in `runall.sh`. 

## Running scaling tests

Scaling tests provide information about how well the different 
distributed strategies scale. We have integrated them into this use case
and you can run them using the `scaling-test.sh` script. 

To generate the plots, refer to the
[Scaling-Test Tutorial](https://github.com/interTwin-eu/itwinai/tree/main/tutorials/distributed-ml/torch-scaling-test#analyze-results).

## Running HPO for EURAC Non-distributed

HPO has been implemented using the Ray tuner to run in a non-distributed 
environment. Refer to `train_hpo.py` file which was adapted from 
`train.py`. The current HPO parameters include learning rate (lr) and batch_size. 

Launch the hpo expirement:

```bash
sbatch startscript_hpo.sh
```

Visualize the HPO results by running `python visualize_hpo.py`. Adjust the `main_dir = '/p/home/jusers/<username>/hdfml/ray_results/<specific run folder name>'` accordingly based on the run folder name, the results path can be got from your slurm output file at the top.

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

> :warning: While the model is loading an error occurs **RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.**
> Possible reasons due to package version mismatch https://github.com/mlflow/mlflow/issues/4903.
