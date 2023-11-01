# PoC for AI-centric digital twin workflows using Singularity containers

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See the latest version of our [docs](https://intertwin-eu.github.io/T6.5-AI-and-ML/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://intertwin-eu.github.io/T6.5-AI-and-ML/docs/How-to-use-this-software.html).


## Requirements

The containers were build using Apptainer version 1.1.8-1.el8 and podman version 4.4.1.

### Base Container

The container are built on top of the [NVIDIA PyTorch NGC Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). The NGC containers come with preinstalled libraries such as CUDA, cuDNN, NCCL, PyTorch, etc that are all harmouniously compatible with each other in order to reduce depenency issue and provide a maximum of portability. The current version used is ```nvcr.io/nvidia/pytorch:23.09-py3```, which is based on CUDA 12.2.1 and PyTorch 2.1.0a0+32f93b1.
If you need other specs you can consults the [Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) and find the right base container for you.


### Running the itwinai container

There are currently three ways to execute the itwinai container on a SLURM cluster.

1. Direct build on the HPC system
2. Use build on the [itwinai repo](https://github.com/interTwin-eu/itwinai/pkgs/container/t6.5-ai-and-ml) and pull to HPC system
3. Deploy to Kubernetes cluster and offload to HPC via [interLink](https://github.com/interTwin-eu/interLink)

![container workflow](docs/docs/img/containers.png) 

#### Direct build
Run the following commands to build the container directly on the HPC system. Select the right base container by altering the following line 
```
apptainer pull itwinai.sif docker://nvcr.io/nvidia/pytorch:23.09-py3
```
inside ```containers/apptainer/apptainer_build.sh``` to change to the desired version.

Install the itwinai libraries by running:
```
./containers/apptainer/apptainer_build.sh
```

Run the startscript with 
```
sbatch use-cases/mnist/torch/startscript.sh
```

#### Github container repository build
With this method you can just pull the ready container from the github container repository:
```
apptainer pull containers/apptainer/itwinai.sif docker://ghcr.io/intertwin-eu/t6.5-ai-and-ml:containers
```

Run the startscript with 
```
sbatch use-cases/mnist/torch/startscript.sh
```

#### InterLink
To be tested


### Future work
It is currently foreseen to build the container via GH actions.