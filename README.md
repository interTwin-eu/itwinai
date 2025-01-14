# itwinai

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai.assess.sqaaas/main/.report/assessment_output.json)

![itwinai Logo](./docs/images/icon-itwinai-orange-black-subtitle.png)

`itwinai` is a powerful Python toolkit designed to help scientists and researchers streamline AI and machine learning
workflows, specifically for digital twin applications. It provides easy-to-use tools for distributed training,
hyper-parameter optimization on HPC systems, and integrated ML logging, reducing engineering overhead and accelerating
research. Developed primarily by CERN, `itwinai` supports modular and reusable ML workflows, with
the flexibility to be extended through third-party plugins, empowering AI-driven scientific research in digital twins.

See the latest version of our docs [here](https://itwinai.readthedocs.io/).

If you are a **developer**, please refer to the [developers installation guide](#installation-for-developers).

## User installation

Requirements:

- Linux or macOS environment. Windows was never tested.

### Python virtual environment

Depending on your environment, there are different ways to
select a specific python version.

#### Laptop or GPU node

If you are working on a laptop
or on a simple on-prem setup, you could consider using
[pyenv](https://github.com/pyenv/pyenv). See the
[installation instructions](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation). If you are using pyenv,
make sure to read [this](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).

#### HPC environment

In HPC systems it is more popular to load dependencies using
Environment Modules or Lmod. If you don't know what modules to load,
contact the system administrator
to learn how to select the proper modules.

##### PyTorch environment

Commands to execute every time **before** installing or activating the python virtual
environment for PyTorch:

- Juelich Supercomputer (JSC):

    ```bash
    ml --force purge
    ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
    ml Python CMake HDF5 PnetCDF libaio mpi4py
    ```

- Vega supercomputer:

    ```bash
    ml --force purge
    ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
    ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
    ```

##### TensorFlow environment

Commands to execute every time **before** installing or activating the python virtual
environment for TensorFlow:

- Juelich Supercomputer (JSC):

    ```bash
    ml --force purge
    ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
    ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12
    ```

- Vega supercomputer:

    ```bash
    ml --force purge
    ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
    ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
    ```

### Install itwinai for users

Install itwinai and its dependencies.

```bash
# First, load the required environment modules, if on an HPC

# Second, create a python virtual environment and activate it
$ python -m venv ENV_NAME
$ source ENV_NAME/bin/activate
```

Install itwinai with support for PyTorch using:

```bash
pip install itwinai[torch]
```

or with TensorFlow support using:

```bash
pip install itwinai[tf]

# Alternatively, if you have access to GPUs
pip install itwinai[tf-cuda]
```

If you want to use Prov4ML logger, you need to install it explicitly since it is only
available on GitHub:

```bash
# For systems with Nvidia GPUs
pip install "prov4ml[nvidia]@git+https://github.com/matbun/ProvML@new-main"

# For MacOs
pip install "prov4ml[apple]@git+https://github.com/matbun/ProvML@new-main"
```

If you also want to install Horovod and Microsoft DeepSpeed for distributed ML with PyTorch,
install them *after* itwinai. You can use this command:

```bash
curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/torch/install-horovod-deepspeed-cuda.sh | bash
```

> [!WARNING]  
> Horovod requires `CMake>=1.13` and
> [other packages](https://horovod.readthedocs.io/en/latest/install_include.html#requirements).
> Make sure to have them installed in your environment before proceeding.

## Installation for developers

If you are contributing to this repository, please continue below for
more advanced instructions.

> [!WARNING]
> Branch protection rules are applied to all branches which names
> match this regex: `[dm][ea][vi]*` . When creating new branches,
> please avoid using names that match that regex, otherwise branch
> protection rules will block direct pushes to that branch.

### Clone the itwinai repository

```bash
git clone [--recurse-submodules] git@github.com:interTwin-eu/itwinai.git
```

### Install itwinai environment

In this project, we are using `uv` as a project-wide package manager. Therefore, if
you are a developer, you should see the [uv tutorial](/docs/uv-tutorial.md) after reading
the following `pip` tutorial.

#### Installation using pip

##### Creating a venv

You can install the `itwinai` environment for development using `pip`. First, however,
you would want to make a Python venv if you haven't already. Make sure you have
Python installed (on HPC you have to load it with `module load Python`), and then you
can create a venv with the following command:

```bash
python -m venv <name-of-venv>
```

For example, if I wanted to create a venv in the directory `.venv` (which is useful if
you use e.g. `uv`), then I would do:

```bash
python -m venv .venv
```

After this you can activate your venv using the following command:

```bash
source .venv/bin/activate
```

Now anything you pip install will be installed in your venv and if you run any python
commands they will use the version from your venv.

##### Installation of packages

We provide some *extras* that can be activated depending on which platform you are
using.

- `dev` for development purposes. Includes libraries for testing and tensorboard etc.
- `torch` for installation with PyTorch.

If you want to install PyTorch using CUDA then you also have to add an
`--extra-index-url` to the CUDA version that you want. Since you are developing the
library, you also want to enable the editable flag, `-e`, so that you don't have to
reinstall everything every time you make a change. If you are on HPC, then you will
usually want to add the `--no-cache-dir` flag to avoid filling up your `~/.cache`
directory, as you can very easily reach your disk quota otherwise. An example of a
complete command for installing as a developer on HPC with CUDA thus becomes:

```bash
pip install -e ".[torch,dev,tf]" \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

If you wanted to install this locally on **macOS** (i.e. without CUDA) with PyTorch, you
would do the following instead:

```bash
pip install -e ".[torch,dev,tf]"
```

If you want to use [Prov4ML](https://github.com/HPCI-Lab/yProvML) logger, you need to install
it explicitly since it is only available on GitHub. Please refer to the
[users installation](#install-itwinai-for-users)
to know more on how to install Prov4ML.

<!-- You can create the Python virtual environments using our predefined Makefile targets. -->

#### Horovod and DeepSpeed

The above does not install `Horovod` and `DeepSpeed`, however, as they require a
specialized [script](env-files/torch/install-horovod-deepspeed-cuda.sh). If you do not
require CUDA, then you can install them using `pip` as follows:

```bash
pip install --no-cache-dir --no-build-isolation git+https://github.com/horovod/horovod.git
pip install --no-cache-dir --no-build-isolation deepspeed
```

#### PyTorch (+ Lightning) virtual environment with makefiles

Makefile targets for environment installation:

- Juelich Supercomputer (JSC): `torch-gpu-jsc`
- Vega supercomputer: `torch-env-vega`
- In any other cases, when CUDA is available: `torch-env`
- In any other cases, when CUDA **NOT** is available (CPU-only installation): `torch-env-cpu`

For instance, on a laptop with a CUDA-compatible GPU you can use:

```bash
make torch-env 
```

When not on an HPC system, you can activate the python environment directly with:

```bash
source .venv-pytorch/bin/activate
```

Otherwise, if you are on an HPC system, please refer to
[this section](#activate-itwinai-environment-on-hpc)
explaining how to load the required environment modules before the python environment.

To  build a Docker image for the pytorch version (need to adapt `TAG`):

```bash
# Local
docker buildx build -t itwinai:TAG -f env-files/torch/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:TAG -f env-files/torch/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai:TAG
```

#### TensorFlow virtual environment

Makefile targets for environment installation:

- Juelich Supercomputer (JSC): `tf-gpu-jsc`
- Vega supercomputer: `tf-env-vega`
- In any other case, when CUDA is available: `tensorflow-env`
- In any other case, when CUDA **NOT** is available (CPU-only installation): `tensorflow-env-cpu`

For instance, on a laptop with a CUDA-compatible GPU you can use:

```bash
make tensorflow-env
```

When not on an HPC system, you can activate the python environment directly with:

```bash
source .venv-tf/bin/activate
```

Otherwise, if you are on an HPC system, please refer to
[this section](#activate-itwinai-environment-on-hpc)
explaining how to load the required environment modules before the python environment.

To  build a Docker image for the tensorflow version (need to adapt `TAG`):

```bash
# Local
docker buildx build -t itwinai:TAG -f env-files/tensorflow/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:TAG -f env-files/tensorflow/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai:TAG
```

### Activate itwinai environment on HPC

Usually, HPC systems organize their software in modules which need to be imported by the users
every time they open a new shell, **before** activating a Python virtual environment.

Below you can find some examples on how to load the correct environment modules on the HPC
systems we are currently working with.

#### Load modules before PyTorch virtual environment

Commands to be executed before activating the python virtual environment:

- Juelich Supercomputer (JSC):

    ```bash
    ml --force purge
    ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
    ml Python CMake HDF5 PnetCDF libaio mpi4py
    ```

- Vega supercomputer:

    ```bash
    ml --force purge
    ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
    ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
    ```

- When not on an HPC: do nothing.

For instance, on JSC you can activate the PyTorch virtual environment in this way:

```bash
# Load environment modules
ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py

# Activate virtual env
source envAI_hdfml/bin/activate
```

#### Load modules before TensorFlow virtual environment

Commands to be executed before activating the python virtual environment:

- Juelich Supercomputer (JSC):

    ```bash
    ml --force purge
    ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
    ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12
    ```

- Vega supercomputer:

    ```bash
    ml --force purge
    ml Python/3.11.5-GCCcore-13.2.0 CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/12.3
    ml GCCcore/11.3.0 NCCL cuDNN/8.9.7.29-CUDA-12.3.0 UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.3.0
    ```

- When not on an HPC: do nothing.

For instance, on JSC you can activate the TensorFlow virtual environment in this way:

```bash
# Load environment modules
ml --force purge
ml Stages/2024 GCC/12.3.0 OpenMPI CUDA/12 MPI-settings/CUDA
ml Python/3.11 HDF5 PnetCDF libaio mpi4py CMake cuDNN/8.9.5.29-CUDA-12

# Activate virtual env
source envAItf_hdfml/bin/activate
```

### Test with `pytest`

Do this only if you are a developer wanting to test your code with pytest.

First, you need to create virtual environments both for torch and tensorflow,
following the instructions above, depending on the system that you are using
(e.g., JSC).

To select the name of the torch and tf environments in which the tests will be
executed you can set the following environment variables.
If these env variables are not set, the testing suite will assume that the
PyTorch environment is under
`.venv-pytorch` and the TensorFlow environment is under `.venv-tf`.

```bash
export TORCH_ENV="my_torch_env"
export TF_ENV="my_tf_env"
```

Functional tests (marked with `pytest.mark.functional`) will be executed under
`/tmp/pytest` location to guarantee isolation among tests.

To run functional tests use:

```bash
pytest -v tests/ -m "functional"
```

> [!NOTE]
> Depending on the system that you are using, we implemented a tailored Makefile
> target to run the test suite on it. Read these instructions until the end!

We provide some Makefile targets to run the whole test suite including unit, integration,
and functional tests. Choose the right target depending on the system that you are using:

Makefile targets:

- Juelich Supercomputer (JSC): `test-jsc`
- In any other case: `test`

For instance, to run the test suite on your laptop user:

```bash
make test
```

## Working with Docker containers

This section is intended for the developers of itwinai and outlines the practices
used to manage container images through GitHub Container Registry (GHCR).

### Terminology Recap

Our container images follow the convention:

```text
ghcr.io/intertwin-eu/IMAGE_NAME:TAG
```

For example, in `ghcr.io/intertwin-eu/itwinai:0.2.2-torch2.6-jammy`:

- `IMAGE_NAME` is `itwinai`
- `TAG` is `0.2.2-torch2.6-jammy`

The `TAG` follows the convention:

```text
[jlab-]X.Y.Z-(torch|tf)x.y-distro
```

Where:

- `X.Y.Z` is the **itwinai version**
- `(torch|tf)` is an exclusive OR between "torch" and "tf". You can pick one or the other, but not both.
- `x.y` is the **version of the ML framework** (e.g., PyTorch or TensorFlow)
- `distro` is the OS distro in the container (e.g., Ubuntu Jammy)
- `jlab-` is prepended to the tag of images including JupyterLab

### Image Names and Their Purpose

We use different image names to group similar images under the same namespace:

- **`itwinai`**: Production images. These should be well-maintained and orderly.
- **`itwinai-dev`**: Development images. Tags can vary, and may include random
hashes.
- **`itwinai-cvmfs`**: Images that need to be made available through CVMFS via
[Unpacker](https://gitlab.cern.ch/unpacked/sync).

> [!WARNING]
> It is very important to keep the number of tags for `itwinai-cvmfs` as low
> as possible. Tags should only be created under this namespace when strictly
> necessary. Otherwise, this could cause issues for the Unpacker.

### Building a new container

Our docker manifests support labels to record provenance information, which can be lately
accessed by `docker inspect IMAGE_NAME:TAG`.

A full example below:

```bash
export BASE_IMG_NAME="what goes after the last FROM"
export IMAGE_FULL_NAME="IMAGE_NAME:TAG"
docker build \
    -t "$IMAGE_FULL_NAME" \
    -f path/to/Dockerfile \
    --build-arg COMMIT_HASH="$(git rev-parse --verify HEAD)" \
    --build-arg BASE_IMG_NAME="$BASE_IMG_NAME" \
    --build-arg BASE_IMG_DIGEST="$(docker pull "$BASE_IMG_NAME" > /dev/null 2>&1 && docker inspect "$BASE_IMG_NAME" --format='{{index .RepoDigests 0}}')" \
    --build-arg ITWINAI_VERSION="$(grep -Po '(?<=^version = ")[^"]*' pyproject.toml)" \
    --build-arg CREATION_DATE="$(date +"%Y-%m-%dT%H:%M:%S%:z")" \
    --build-arg IMAGE_FULL_NAME=$IMAGE_FULL_NAME \ 
    .
```
