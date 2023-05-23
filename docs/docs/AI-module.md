---
layout: default
title: AI module
nav_order: 4
---

# AI module

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

![image](img/user-platform%20interaction.png)

### `itwinai` - DTE's AI module

Below we present the planned improvements and features for versions of the AI module. At the end of each release,
we plan to validate the prototype by deploying it on an HPC centre (e.g., FZJ).

#### `itwinai v0.0` - Minimal working baseline (MWB)

- Input dataset path
- Pytorch Lightning (PL) Model name: already available inside `itwinai` module. Just need to be imported
- Simple numeric hyperparams: learning rate, batch size
- MNIST use case
- No logs are saved

#### `itwinai v0.1` - Consolidated AI training

- Output logs path/URI
- Mlflow: logs saved to local filesystem
  - Log metrics (metrics logger)
  - Save model parameters
- Hyperparams:
  - Optimizer name (standard only, to be imported form torch)
  - Scheduler name (standard only, to be imported form torch)
  - PL callbacks

#### `itwinai v0.2` - Import custom functionalities

- Path to user-provided custom python script
- Custom PL model name: has to be imported from user-provided python script
- Name of custom PL Data model, provided by the user
- Mlflow save model to models registry
- TensorFlow support

#### `itwinai v0.3` - HPC support

- Distributed ML
  - Torch DDP
  - Horovod
- Containers and integration with workflow engine (e.g., Apache airflow)

#### `itwinai v0.4` - Tune

- Ray tune (needs Ray cluster on the infra)

#### `itwinai v0.5` - Kubernetes

Needs Kubernetes on the infra

- Mlflow full deployment:
  - SQL service
  - S3 object storage service

To be allocated:

- Online ML
