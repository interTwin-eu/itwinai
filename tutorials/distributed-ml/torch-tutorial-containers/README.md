# Using itwinai container to run training on MNIST

**Author(s)**: Matteo Bunino (CERN)

In this tutorial we show how to use containers to run machine learning workflows with itwinai.

Container images are pulled from the GHCR associated with our GH repository and are Docker
images. Although Docker is generally not supported in HPC environments, Singularity supports
Docker images and is able to convert them to Singularity images (SIF files) upon pull.

The examples will showcase training of a simple neural network defined in `model.py` on the
MNIST benchmark dataset. The ML workflow is defined using itwinai `Pipeline`, the training
algorithm is implemented by the itwinai `TorchTrainer`, and the training parameters are
defined in `config.yaml`.

In this tutorial we are using general purpose itwinai container images to execute a use case code.
This is possible when the use case does not depend on additional packages not included in the container
image. If you want to add dependencies, you need to create a new container image using itwinai as
base image. A minimal example of a custom Dockerfile:

```dockerfile
FROM ghcr.io/intertwin-eu/itwinai:0.2.2-torch-2.1
RUN pip install --no-cache-dir PYTHON_PACKAGE
```

## Docker (non-HPC environments)

When executing a Docker container, you need to explicitly mount the current working directory
in the container, making it possible for the script executed in the container to use existing
files and create new files in the current directory (on in another location). This can be achieved
by bind mounting the current working directory in some location in the container, and moving to
that location in the container before executing the desired command.

```bash
bash run_docker.sh
```

The script above runs the following command in the itwinai torch container
in this folder:

```bash
itwinai exec-pipeline +pipe_key=training_pipeline
```

> [!WARNING]
> When using Docker, if your container does not recognizes the GPUs of your VM
> you may need to install the
> [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
> , if not already installed.

## Singularity (HPC environments)

With singularity there is no need to explicitly bind mount the current working directory (CWD) in the container
as this is already done automatically by Singularity. Moreover, the CWD inside the container *coincides*
with the CWD outside the container, not requiring to change directory before executing the command inside
the container. However, differently from Docker, Singularity does
not automatically allow to write in locations inside the container. It is therefore suggested to save
results in the CWD, or in other locations mounted in the container.

First of all, pull the Docker image and convert it to a Singularity image:

```bash
# If needed, remove existing Singularity image before proceeding
rm -rf itwinai_torch.sif

# Pull Docker image and convert it to Singularity on login node
singularity pull itwinai_torch.sif docker://ghcr.io/intertwin-eu/itwinai:0.2.2-torch-2.1
```

Before running distributed ML on the computing node of some HPC cluster, make sure to download
the dataset as usually there is not internet connection on compute nodes:

```bash
# Run only the first step on the HPC login node, which downloads the datasets if not present
singularity run itwinai_torch.sif /bin/bash -c \
    "itwinai exec-pipeline +pipe_key=training_pipeline +pipe_steps=[dataloading_step]"
```

Now run distributed ML on multiple compute nodes using both Torch DDP and Microsoft DeepSpeed:

```bash
# Run on distributed ML job (torch DDP is the default one)
sbatch slurm.sh

# Alternatively, run all distributed jobs
bash runall.sh
```

> [!NOTE]
> Please note that at the moment Horovod distributed training using containerized environments
> is not supported.
