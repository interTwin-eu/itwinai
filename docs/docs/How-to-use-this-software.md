---
layout: default
title: How to use this software
nav_order: 2
---

# How to use this software
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

This guide provides a detailed explanation on how to use the AI/ML workflow
tool, developed in the context of [interTwin](https://github.com/interTwin-eu/).

**Target audience**: anyone aiming to simplify MLOps for their digital twin (DT)
use case/project. Use cases from interTwin project.

## Clone this repo

```bash
git clone git@github.com:interTwin-eu/T6.5-AI-and-ML.git
```

A new use case/project can be added under `./use-cases` folder.

Build the workflow runner environment and development environment
following the instructions on the README file.

## Define a DT workflow

Before delving into workflow definition rules, make sure to have
understood *what is* a [workflow](./Concepts#workflow) in this context.

You can define one or more workflows for your DT use case (e.g., ML training,
ML inference, other). A workflow is
defined through configuration files in the use case subfolder.
For the same use case, a DT developer can define multiple workflows,
in which multiple datasets are involved.

Currently, each step is executed in an isolated Python virtual environment,
built according to [conda](https://docs.conda.io/en/latest/) standards.
In practice, it is built with
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html),
which is faster.

To begin with, you can start by looking at an example of the
[MNIST toy use case](use-cases/mnist), located at `use-cases/mnist`
in the code repository.

### Use case metadata

The main configuration file of an use case is `meta.yml`, which stores
the metadata of it. When creating a new use case, you need to update the
`root` field with the path to the use case folder, with respect to the
repo root.

The datasets registry is a field in this configuration file,
which stores the metadata
for all datasets involved in a use case. This configuration provides a
unified place where datasets can be maintained, making it easy to access
them from other configuration files.

The dataset registry has the format:

```yaml
datasets:
  some-dataset-name:
    doc: Documentation string for this dataset
    location: path/to/dataset/disk/location
```

Example of `meta.yml` from [MNIST use case](use-cases/mnist):

```yaml
# Use case root location. End without path '/' char!
root: ./use-cases/mnist

# AI folder location. End without path '/' char!
ai-root: ./ai

# Datasets registry
datasets:
  preproc-images:
    doc: Preprocessed MNIST images
    location: ${root}/data/preproc-images
  ml-logs:
    doc: MLflow tracking URI for local logging
    location: ${root}/data/ml-logs
  ml-predictions:
    doc: predictions on unseen data
    location: ${root}/data/ml-predictions
```

Datasets are imported from the datasets registry to other files by means
of [OmegaConf](https://omegaconf.readthedocs.io/)'s
[variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation).
This way, you can easily import datasets metadata (e.g., location on
file system) from datasets registry.

Dataset registry of an use case can be visualized using [itwinai CLI](./CLI#visualization):

```bash
USE_CASE_ROOT='use-cases/mnist/'
micromamba activate ./.venv-dev && \
  itwinai datasets --use-case $USE_CASE_ROOT
```

### Workflow configuration

Use case workflows are defined with `*-workflow.yml` files in the use case root,
and there are two ways to define a workflow:

- "Custom" format for workflow definition is an intuitive standard we created
for this prototype, for easy prototyping.
- [Common Workflow Language](https://www.commonwl.org/) (CWL), which is
currently under development, and not ready to be used.

Which of the two is used is defined by setting the `--cwl` flag (explained
[below](#run-the-workflow)).

#### Custom workflow definition

To define a workflow with the custom format, the DT developer must follow
the structure provided below.

The `steps` section defines the steps of the workflow, in the order in which
they have to be executed:

```yaml
steps:
  - some-step-name:
      doc: Documentation string for this step
      env: # micromamba environment metadata
        file: some-conda-env.yml
        prefix: path/to/conda/env/
      command: Command to execute inside micromamba env
      args: # Command arguments.
        # Note interpolation with datasets registry here! 
        some-arg: ${datasets.my-dataset.location}
        some-other-arg: 42
  - next-step-name:
    ...
```

Example workflow from [MNIST use case](use-cases/mnist), defined in
`training-workflow.yml`:

```yaml
steps:
  - preprocessing:
      doc: Download and split MNIST dataset into train and test sets
      command: python ${root}/mnist-preproc.py
      env: 
        file: ${root}/env-files/preproc-env.yml
        prefix: ${root}/.venv-preproc
      args:
        output: ${datasets.preproc-images.location}
        stage: train
  - ml-training:
      doc: Train a neural network to classify MNIST images
      command: itwinai train
      env: 
        file: ${ai-root}/env-files/pytorch-lock.yml
        prefix: ${ai-root}/.venv-pytorch
        source: ${ai-root}
      args:
        train-dataset: ${datasets.preproc-images.location}
        ml-logs: ${datasets.ml-logs.location}
        config: ${root}/mnist-ai-train.yml
```

Step 1 is named `preprocessing` and uses the `mnist-preproc.py` script to pre-process the MNIST dataset. It takes no
input, generates an output dataset named `preproc-images`, and uses an environment defined in a YAML file named
`preproc-env.yml` located in the `./use-cases/mnist` directory.
Step 2 is named `ml-training` and trains a machine learning model using the preprocessed image data generated in
the first step. The training is performed using the train command from the `itwinai` tool. The input dataset is
`preproc-images`, and the output is `ml-logs`. The step uses an environment defined in a YAML file named
`pytorch-env.yml` located in the `./ai` directory. The machine learning model is configured using the `mnist-ai-train.yml`
file located in the `./use-cases/mnist` directory.

#### CWL: under development and not ready to be used yet

**NOTE**. At the moment, support for CWL is under development,
and is not available.

<!-- The CWL style workflow is defined in the bottom of the `training-workflow.yml` file:

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
the `./ai` directory. The `trainingCommand` section defines the command to run the training. -->

## Implement workflow steps

Implement use case-specific steps.
Note that the implementation of steps involving AI/ML are addressed in the next
step, and they can be implemented a bit more easily.

Each step of a workflow is characterized by its python virtual environment
and a command to be executed in that
environment. A command can be implemented by providing, for instance, a python script to be executed in some environment.

To execute a step, the workflow engine will run something like:

```bash
micromamba run -p PATH_TO_STEP_ENV CMD --arg1 ARG_1_VAL ... --argN ARG_N_VAL
```

Where:

- `PATH_TO_STEP_ENV` is the path to the micromamba environment for this step.
- `CMD` is the command to execute in that environment.
- The developer can use additional parameters which are automatically appended
to the command.

*Example*: in the [MNIST use case](use-cases/mnist),
the preprocessing step is implemented by a python script, which downloads and
splits the MNIST dataset in a specific location. Using a command similar to:

```bash
micromamba run -p ./use-cases/mnist/.venv-preproc \
    python ./use-cases/mnist/mnist-preproc.py \
    --output ./data/mnist/preproc-images-inference \
    --stage train
```

## Define AI/ML workflow

AI/ML workflows are implemented by the `itwinai` module.
The DT developer, who wants to include a new use case, needs to provide
only a reduced amount of code to describe a neural network, plus some
configuration files.

The developer must implement the neural network to train and include it inside
`itwinai` python package, under `ai/src/itwinai`. For instance, under
`ai/src/itwinai/plmodels` when using PyTorch Lightning.

For instance, `LitMNIST` neural network used in [MNIST use case](use-cases/mnist)
has been added under `ai/src/itwinai/plmodels/mnist.py`

Once a model has been included inside the `itwinai` python module, it can be imported during training.
In the future, `itwinai` will support also neural networks not provided out-of-the-box by `itwinai`.

The developer must define two configuration files to access `itwinai`
functionalities.
First, ML training configuration, associated with `$ itwinai train` [CLI](./CLI) command.
Second, ML inference configuration, associated with `$ itwinai predict` [CLI](./CLI) command.

MLOps heavily relies on commands provided by [itwinai CLI](./CLI).
Therefore, before continuing, make sure to have understood how
[itwinai CLI](./CLI) works.

### ML training configuration

ML training configuration is provided in a with naming convention `*-ai-train.yml`
under the use case root directory.

An example configuration file is provided below, where the fields have been replaced with their respective description:

```yaml
# Configuration file of AI workflows for X use case

# Training configuration
train:
  type: >
    can be 'lightning' or 'tf', depending whether the neural network is defined
    using PyTorch Lightning or TensorFlow.
    At the moment, only 'lightning' is supported.
  
  # Configuration format defined by PyTorch Lightning CLI
  # https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html
  conf:
    # See discussion below
    ...

# MLFlow logger configuration
logger:
  experiment_name: >
    Unique name for an experiment, to group all similar
    runs under the same experiment
  description: Description for this specific run.
  log_every_n_epoch: how often to log (epochs)
  log_every_n_steps: how often to log (steps, i.e., batches)
  registered_model_name: >
    Unique name used in Models Registry to identify an ML model.
    If given, it is automatically registered in the Models Registry.
```

When using PyTorch Lightning (PL) ML framework, the training configuration is easy to define, as it follows the schema
pre-defined by lightning authors for the PL CLI. See its documentation
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#trainer-callbacks-and-arguments-with-class-type),
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#trainer-callbacks-and-arguments-with-class-type),
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#multiple-models-and-or-datasets), and
[here](https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#optimizers-and-learning-rate-schedulers).

An example taken from
[MNIST use case](use-cases/mnist) located at `use-cases/mnist/mnist-ai-training.yml`:

```yaml
# Pytorch lightning config for training
train:
  type: lightning
  # Follows lightning config file format:
  # https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html#multiple-models-and-or-datasets
  conf:
    seed_everything: 4231162351

    # Lightning Trainer configuration
    trainer:
      accelerator: auto
      strategy: auto
      devices: auto
      num_nodes: 1
      precision: 32-true
      
      # MLFlow logger (initial) configuration.
      # To be completed with run details, later on
      logger:
        class_path: lightning.pytorch.loggers.MLFlowLogger
        init_args:
          experiment_name: ${logger.experiment_name}
          save_dir: ./mlruns
      
      # Callbacks
      callbacks:
        - class_path: lightning.pytorch.callbacks.early_stopping.EarlyStopping
          init_args:
            monitor: val_loss
            patience: 2
        - class_path: lightning.pytorch.callbacks.lr_monitor.LearningRateMonitor
          init_args:
            logging_interval: step
        - class_path: lightning.pytorch.callbacks.ModelCheckpoint
          init_args:
            dirpath: checkpoints
            filename: best-checkpoint
            save_top_k: 1
            verbose: true
            monitor: val_loss
            mode: min

      max_epochs: 1
      
    # Lightning Model configuration
    model:
      class_path: itwinai.plmodels.mnist.LitMNIST
      init_args:
        hidden_size: 64

    # Lightning data module configuration
    data:
      class_path: itwinai.plmodels.mnist.MNISTDataModule
      init_args:
        data_dir: ${cli.train_dataset}
        batch_size: 32

    # Torch Optimizer configuration
    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001

    # Torch LR scheduler configuration
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.ExponentialLR
      init_args:
        gamma: 0.1

# Mlflow
logger:
  experiment_name: MNIST classification lite
  description: A MLP classifier for MNIST dataset.
  log_every_n_epoch: 1
  log_every_n_steps: 1
  # Name used in Models Registry. If given, it is automatically
  # registered in the Models Registry.
  registered_model_name: MNIST-clf-lite
```

Note the field `data_dir: ${cli.train_dataset}` in the above configuration.
More on this later.

### ML inference configuration

ML training configuration is provided in a with naming convention
`*-ai-inference.yml` under the use case root directory.

An example configuration file is provided below, where the fields have been replaced with their respective description:

```yaml
inference:
  experiment_name: >
    Unique name for an experiment, to group all similar
    runs under the same experiment
  run_id: Run ID in MLFlow server of pre-trained model
  ckpt_path: model/checkpoints/best-checkpoint/best-checkpoint.ckpt
  train_config_artifact_path: name of training config saved to MLFlow artifacts folder
  type: >
    can be 'lightning' or 'tf', depending whether the neural network is defined
    using PyTorch Lightning or TensorFlow.
    At the moment, only 'lightning' is supported.
  
  # Configuration format defined by PyTorch Lightning CLI
  # https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html
  conf:
    # See discussion below
    ...
```

Regarding the `inference.conf` field, same considerations hold as for `train.conf` field of ML training configuration.

An example taken from
[MNIST use case](use-cases/mnist) located at `use-cases/mnist/mnist-ai-training.yml`:

```yaml
inference:
  type: lightning
  experiment_name: MNIST classification lite
  # Run ID in MLFlow server: pre-trained model
  run_id: 54f790100be646e0a7ccbb1235729d00
  ckpt_path: model/checkpoints/best-checkpoint/best-checkpoint.ckpt
  train_config_artifact_path: pl-training.yml
  conf:
    # Lightning data module configuration
    data:
      class_path: itwinai.plmodels.mnist.MNISTDataModule
      init_args:
        data_dir: ${cli.input_dataset}
        batch_size: 32
```

### Accessing CLI args from config file

As explained above, train and predict commands in itwinai CLI receive as input
specific configuration files:

- The `train` command receives `*-ai-train.yml` as configuration.
- The `predict` command receives `*-ai-inference.yml` as configuration.

With [OmegaConf](https://omegaconf.readthedocs.io/)'s
[variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation)
you can access the args from the itwinai CLI command from the configuration file
associated with this command.

Example: the field `data_dir: ${cli.input_dataset}` in the above configuration
accesses the value of `--input-dataset` argument of `itwinai predict` command.

### ML framework: PyTorch vs. TensorFlow

At the moment, only PyTorch are supported. TensorFlow support is planned for
future releases.

## Run the workflow

Once a workflow has been configured, it can be run by executing `run-workflow.py` in the root of this repo:

```bash
micromamba run -p ./.venv python run-workflow.py -f WORKFLOW_DEFINITION_FILE
```

This script performs two main actions:

1. Deploy ste steps of a workflow as python environments, managed with Conda.
2. Run a workflow step-by-step, following the directives given in the config file.

See some examples of workflow executions in `examples.sh`, for instance:

```bash
# Run workflow for MNIST toy use case
micromamba run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml
```

<!-- If you want to run using the Common Workflow Language (CWL).
Note that logging and saving models/metric is currently not supported using CWL.

A workflow can be executed following CWL definition, adding the `--cwl` flag to the command above.

**NOTE**: CWL support is still experimental and may not be fully working.

```bash
micromamba run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml --cwl
``` -->

## Write tests cases

Integrating an new use case means defining new workflows for it.
It is strongly suggested to define "integration" test cases for
those workflows. This way, every time `itwinai`
framework is updated, integration tests automatically verify that
the use case integrates well with the new changes introduced in the
main framework.
Moreover, integration tests verify that an use case case is stable,
and is not hiding some "bug".

Add test for your use case under the `test/` folder. You can take
inspiration from other use cases' tests.
