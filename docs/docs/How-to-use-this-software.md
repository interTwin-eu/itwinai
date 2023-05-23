---
layout: default
title: How to use this software
nav_order: 2
---

This guide provides a detailed explanation on how to use the AI/ML workflow tool, developed in the context of interTwin.

**Target audience**: anyone aiming to simplify MLOps for their digital twin (DT) use case/project.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

## 1. Clone this repo

```bash
git clone git@github.com:interTwin-eu/T6.5-AI-and-ML.git
```

A new use case/project can be added under `./use-cases` folder.

Build the workflow runner environment:

```bash
# This will create a local conda environment, located at ./.venv
make
```

## 2. Define a DT workflow

You can define one or more workflows for your DT use case (e.g., ML training, ML inference, other). A workflow is
defined through configuration files in the use case subfolder.

To begin with, you can start by looking at an example of the [MNIST toy use case](https://github.com/interTwin-eu/T6.5-AI-and-ML/tree/main/use-cases/mnist).

The workflow is defined in the `training-workflow.yml` file and is written in two different styles: classical style and
CWL style. Which of the two is used is defined by setting the `--cwl` flag (explained [below](#5-run-the-workflow)).
The classical style is described in the top and is composed of the `datasets` and `steps` sections:

```yaml
#Workflow definition in classical style
datasets:
  preproc-images:
    location: ./data/mnist/preproc-images
  ml-logs:
    location: ./data/mnist/ml-logs
```

The datasets section defines the input and output datasets of the workflow. There are two datasets defined in this
workflow: `preproc-images` and `ml-logs`.
The `preproc-images` dataset is the output of the preprocessing step and contains preprocessed image data for training.
The `ml-logs` dataset contains logs generated during machine learning training.

The `steps` section defines the two steps of the workflow:

```yaml
steps:
  - preprocessing:
      input: null
      output: preproc-images
      command: python ./use-cases/mnist/mnist-preproc.py
      env: 
        file: ./use-cases/mnist/preproc-env.yml
        prefix: ./use-cases/mnist/.venv-preproc
        source: null
      params: null
  - ml-training:
      input: preproc-images
      output: ml-logs
      # command: python ./ai/ai-training.py
      command: itwinai train
      env: 
        file: ./ai/pytorch-env.yml
        prefix: ./ai/.venv-pytorch
        source: ./ai
      params:
        config: ./use-cases/mnist/mnist-ai.yml
```

Step 1 is named `Preprocessing` and uses the `mnist-preproc.py` script to preprocess the MNIST dataset. It takes no
input, generates an output dataset named `preproc-images`, and uses an environment defined in a YAML file named
`preproc-env.yml` located in the `./use-cases/mnist` directory.
Step 2 is named `ml-training` and and trains a machine learning model using the preprocessed image data generated in
the first step. The training is performed using the train command from the `itwinai` tool. The input dataset is
`preproc-images`, and the output is `ml-logs`. The step uses an environment defined in a YAML file named
`pytorch-env.yml` located in the `./ai` directory. The machine learning model is configured using the `mnist-ai.yml`
file located in the `./use-cases/mnist` directory.

The CWL style workflow is defined in the bottom of the `training-workflow.yml` file:

```yaml
#Workflow definition in CWL style

workflowFileCWL: use-cases/mnist/workflow.cwl

preprocessEnvironment:
  class: Directory
  path: .venv-preproc

preprocessScript:
  class: File
  path: mnist-preproc.py

#Placeholder for input dataset
# preprocessInput:
#   class: Directory
#   path:

preprocessOutput:
  class: Directory
  path: ../../data/mnist/preproc-images

#training step
trainingConfig:
  class: File
  path: mnist-ai.yml

trainingEnvironment:
  class: Directory
  path: ../../ai/.venv-pytorch

trainingCommand: train
```

The `workflowFileCWL` section defines the path to the CWL file that contains the workflow definition. The
`preprocessEnvironment` section defines the environment required to run the `mnist-preproc.py` script.
It is a directory named `.venv-preproc`. The `preprocessScript` section defines the path to the script.
Training
The `trainingConfig` section defines the configuration for the machine learning model. It is a file named
`mnist-ai.yml` located in the `./use-cases/mnist` directory. The `trainingEnvironment` section defines the
environment required to run the machine learning training. It is a directory named `.venv-pytorch` located in
the `./ai` directory. The `trainingCommand` section defines the command to run the training.

## 3. Develop each step

Implement use case-specific steps. Note that AI/ML workflows will be addressed in the next step, and they can be
implemented a bit more easily.

Each step of a workflow is characterized by its python virtual environment (PVE) and a command to be executed in that
environment. A command can be implemented by providing, for instance, a python script to be executed in some PVE.

To execute a step, the workflow engine will run something like:

```bash
conda run -p PATH_TO_STEP_ENV CMD --input INPUT --output OUTPUT [--param1 PARAM_1_VAL ... --paramN PARAM_N_VAL]
```

Where:

- `PATH_TO_STEP_ENV` is the path to the PVE (conda) for this step.
- `CMD` is the command to execute in that PVE.
- `INPUT` is the path to the input dataset for this step.
- `OUTPUT` is the path to where to write the output of this step.
- The developer can use additional parameters to enrich the input of the workflow step

*Example*: in the MNIST toy use case, the preprocessing step is implemented by a python script, which downloads and
splits the MNIST dataset in a specific location. Using a command similar to:

```bash
conda run -p ./use-cases/mnist/.venv-preproc \
    python ./use-cases/mnist/mnist-preproc.py \
    --input null \
    --output ./data/mnist/preproc-images-inference
```

## 4. Define AI/ML workflow

AI/ML workflows are implemented by the `itwinai` module. The DT developer who wants to include a new use case has to provide
only a reduced amount of code to describe its neural network, plus some configuration files.

The developer must implement the neural network she wants to train and include it inside
[`itwinai`](../../ai/src/itwinai) module. For instance, see the example of
[LitMNIST model](../../ai/src/itwinai/plmodels/mnist.py#L15).
Once a model has been included inside the `itwinai` python module, then it can be imported during training.
In the future, `itwinai` will support also neural networks not provided out-of-the-box by `itwinai`.

The developer must define two configuration files to access `itwinai` functionalities:

1. One for high-level definition of ML workflow.
1. Another specific for ML training, or inference.

### 1. MLOps configuration

First of all, the developer must provide some high-level MLOps configuration for the use case she wants to integrate.
An example configuration file is provided below, where the fields have been replaced with their respective description:

```yaml
# Configuration file of AI workflows for X use case

# Training configuration
train:
  type: can be 'lightning' or 'tf', depending whether the neural network is defined in PyTorch Lightning or TensorFlow. At the moment, only 'lightning' is supported.
  path: path to the training configuration (YAML) file. It is in a format coherent with the type of training framework of choice (i.e., lightning or tf).

inference:
  type: lightning
  # MLFlow tracking URI
  tracking_uri: file:./data/mnist/ml-logs
  # Run ID in MLFlow server: pre-trained model
  # IMPORTANT: this has to be updated!
  run_id: e1383711ab42434eb69d67cd281fbf76
  ckpt_path: model/checkpoints/best-checkpoint/best-checkpoint.ckpt
  train_config_artifact_path: pl-training.yml 

# Mlflow logger configuration
logger:
  experiment_name: Unique name for an experiment, to group all similar runs under the same experiment
  description: Description for this specific run.
  log_every_n_epoch: how often to log (epochs)
  log_every_n_steps: how often to log (steps, i.e., batches)
  registered_model_name: Unique name used in Models Registry to identify an ML model. If given, it is automatically registered in the Models Registry.
```

### 2. Framework-specific configuration

Depending on whether you are using PyTorch Lightning or TensorFlow, you may need to provide a different ML configuration.

#### PyTorch Lightning

When using PyTorch Lightning (PL) ML framework, the training configuration is easy to define, as it follows the schema
pre-defined by lightning authors for the PL CLI. See its documentation
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#trainer-callbacks-and-arguments-with-class-type),
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#trainer-callbacks-and-arguments-with-class-type),
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#multiple-models-and-or-datasets), and
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#optimizers-and-learning-rate-schedulers).

An example taken from
[MNIST use case](https://github.com/interTwin-eu/T6.5-AI-and-ML/tree/main/use-cases/mnist):

```yaml
# lightning.pytorch==2.0.1.post0
seed_everything: 4231162351

# Lightning Trainer configuration
trainer:
  accelerator: auto # E.g., 'çpu', 'gpu' 
  devices: auto # Which devices to use (e.g., GPU number)
  num_nodes: 1 # Number of computing nodes (distributed training)
  
  # MLFlow logger (initial) configuration.
  # To be completed with run details, later on
  logger:
    class_path: lightning.pytorch.loggers.MLFlowLogger
    init_args:
      experiment_name: lightning_logs
      save_dir: ./mlruns 
  
  # Callbacks: some of the most interesting out-of-the-box capabilities from PyTorch Lightning
  callbacks:
    - class_path: lightning.pytorch.callbacks.early_stopping.EarlyStopping
      init_args:
        monitor: val_loss
        patience: 2
    - class_path: lightning.pytorch.callbacks.lr_monitor.LearningRateMonitor
      init_args:
        logging_interval: step

  max_epochs: 5 # Max number of training epochs


# Configuration for Lightning Model subclass: the model to train
model:
  class_path: itwinai.plmodels.mnist.LitMNIST # Path to the class of the ML model (PyTorch Lightning) to train 
  # Constructor args for itwinai.plmodels.mnist.LitMNIST
  init_args:
    data_dir: ./data/mnist/preproc-images
    hidden_size: 64
    learning_rate: 0.0002
    batch_size: 32

# Torch Optimizer configuration: choose optimizer and define its hyper-parameters
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.001 # Overrides learning_rate argument provided for itwinai.plmodels.mnist.LitMNIST constructor argument 

# Torch LR scheduler configuration: choose learning rate scheduler
lr_scheduler:
  class_path: torch.optim.lr_scheduler.ExponentialLR
  init_args:
    gamma: 0.1
```

#### TensorFlow

At the moment, TensorFlow models are not supported.

TODO: document configuration files for TensorFlow

## 5. Run the workflow

Once a workflow has been configured, it can be run by executing `run-workflow.py` in the root of this repo:

```bash
conda run -p ./.venv python run-workflow.py -f WORKFLOW_DEFINITION_FILE
```

This script performs two main actions:

1. Deploy ste steps of a workflow as python environments, managed with Conda.
2. Run a workflow step-by-step, following the directives given in the config file.

A workflow can be executed following CWL definition, adding the `--cwl` flag to the command above.

See some examples of workflow executions in
[`examples.sh`](https://github.com/interTwin-eu/T6.5-AI-and-ML/blob/main/examples.sh) ,
for instance:

```bash
# Run workflow for MNIST toy use case
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml
```

If you want to run using the Common Workflow Language (CWL).
Note that logging and saving models/metric is currently not supported using CWL.

**NOTE**: CWL support is stil experimental and may not be fully working.

```bash
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml --cwl
```
