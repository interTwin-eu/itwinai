---
layout: default
title: MNIST
parent: Use cases
nav_order: 1
---

# MNIST: toy example for DT workflows
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

Of course MNIST images classification is not a digital twin. Still, it is useful to
provide an example on how to define an end-to-end digital twin workflow with the
software provided in this repository.

The MNIST use case implements two workflows:

1. Training workflow: train a neural network to classify MNIST images, and save the trained
neural network to the Models Registry.

    ```mermaid
    flowchart LR
        %% Nodes
        preproc(Pre-processing)
        ai(ML training)
        reg[(Models Registry:\npre-trained ML models)]

        %% Workflow
        preproc --> ai

        %% Connections
        ai -.-> |Saves to| reg
    ```

    This workflow is executed by running the command:

    ```bash
    conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml
    ```

1. Inference workflow: use the pre-trained neural network to classify unseen images (the test set, in this case).

    ```mermaid
        flowchart LR
            %% Nodes
            preproc(Pre-processing)
            ai_depl(ML inference)
            pred[(Predictions)]

            %% Workflow
            preproc --> ai_depl 

            %% Connections
            ai_depl -.-> |Saves to| pred
    ```

    This workflow is executed by running the command:

    ```bash
    conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/inference-workflow.yml
    ```

The interactions among workflows and their steps can be described in more details as the following, where conceptual ordering
among different workflow steps is represented by solid arrows:

```mermaid
graph TD
    %% Nodes
    remote_repo[(Remote repo)]
    preproc(Pre-processing)
    ai(ML training)
    ai_depl(ML inference)
    train_set[(Train dataset)]
    test_set[(Test dataset)]
    ml_logs[(ML logs)]
    reg[(Models Registry:\npre-trained ML models)]
    pred[(Predictions)]

    %% Workflow
    preproc --> ai ---> ai_depl

    %% Connections
    preproc -.-> |Fetches| remote_repo
    preproc -.-> |Stores| train_set
    preproc -.-> |Stores| test_set
    ai -.-> |Trains/validates model| train_set
    ai -.-> |Tests model| test_set
    ai -.-> |Stores model to| reg
    ai -.-> |Logs| ml_logs
    ai_depl -.-> |Loads from| reg
    ai_depl -.-> |Predict from| test_set
    ai_depl -.-> |Stores| pred
```

## Workflow steps

Here we explain in more details how the workflow steps have been configured.
Configuration files and Python scripts are organized under `use-cases/mnist/`
folder, in the core repository.

### Pre-processing

This step is implemented by executing `mnist-preproc.py` script in its dedicated conda environment, defined by
`preproc-env.yml`. This solution gives full freedom to the DT developer to implement any preprocessing logic, adaptable
to any custom dataset format.

### ML training

Is the step in which a neural network is trained on the training dataset, and
validated on the validation dataset.
The mentioned datasets are a result from a split of the pre-processed training
dataset, produced by the pre-processing step.
This step completes the **training workflow**, and results into ML logs and a
trained neural network, which is saved to
the Models Registry. The training workflow can be re-run multiple times with different (hyper)parameters, with the goal
of optimizing some ML validation metric. The neural network with the best validation performances is used to make
predictions on unseen data, in the inference step.

ML training logic is implemented by the `itwinai` library, requiring the DT developer to produce only a set of YAML
configuration files. For the moment, we assume that the neural network and the training code is already present in
`itwinai` library.

Both ML training and inference are implemented by commands executed in the same virtual environment. At the moment,
only PyTorch is supported. The corresponding virtual environment definition, used by the `itwinai` library,
is located at `ai/pytorch-env.yml`.

The ML training logic provided by `itwinai` library is accessed via the
[itwinai CLI](../CLI).

The DT developer must provide a training configuration file, following some
rules explained in [this section](../How-to-use-this-software#ml-training-configuration). For MNIST use case, the
training configuration is provided by `mnist-ai-train.yml` file.

Training command is automatically managed by the workflow runner, but it can also
be triggered from withing the ai environment running the following command:

```bash
conda activate ./ai/.venv-pytorch && \
    itwinai train --train-dataset $TRAINING_DATASET_PATH \
        --ml-logs $MLFLOW_TRACKING_URI \
        --config ./use-cases/mnist/mnist-ai-train.yml
```

While training is running, the produced ML logs can be inspected in real-time from MLFlow UI by running the command in
the training virtual environment (Conda):

```bash
conda activate ./ai/.venv-pytorch && \
    itwinai mlflow-ui --path $PATH_TO_ML_LOGS
```

### ML inference

A pre-trained neural network is applied to a set of data which was not used to train it. In fact, this is defined as
"unseen" data, from the neural network perspective. An example of ML inference is the application of a trained neural
network to make predictions on new data, to support decision making. *Example*: forecast fire risk maps in Sicily in
August 2023, starting from newly-collected satellite images, to alert local authorities in case of elevated fire risk.

To select a pre-trained ML model, the DT developer must retrieve the `RUN ID` of
the training run created by MLFLow for some specific training.

The, the DT developer can update the inference configuration file
`mnist-ai-inference.yml` and run inference workflow.

Inference/prediction command is automatically managed by the workflow runner, but it can also be triggered from within
the ai environment running the following command:

```bash
conda activate ./ai/.venv-pytorch && \
    itwinai predict --input-dataset $UNSEEN_EXAMPLES_DATASET_PATH \
        --predictions-location $PREDICTIONS_LOCATION \
        --ml-logs $PATH_TO_ML_LOGS \
        --config ./use-cases/mnist/mnist-ai-inference.yml 
```

## References

To learn more on how to use this software, e.g., to deploy a new use case, please refer to [this guide](../How-to-use-this-software).
