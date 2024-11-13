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

- [User installation](#user-installation)
  - [Python virtual environment](#python-virtual-environment)
    - [Laptop or GPU node](#laptop-or-gpu-node)
    - [HPC environment](#hpc-environment)
      - [PyTorch environment](#pytorch-environment)
      - [TensorFlow environment](#tensorflow-environment)
  - [Install itwinai for users](#install-itwinai-for-users)
- [Installation for developers](#installation-for-developers)
  - [Clone the itwinai repository](#clone-the-itwinai-repository)
  - [Install itwinai environment](#install-itwinai-environment)
    - [Installation with pip or uv](#installation-with-pip-or-uv)
      - [Installation using `uv.lock`](#installation-using-uvlock)
    - [PyTorch (+ Lightning) virtual environment](#pytorch--lightning-virtual-environment)
    - [TensorFlow virtual environment](#tensorflow-virtual-environment)
  - [Activate itwinai environment on HPC](#activate-itwinai-environment-on-hpc)
    - [Load modules before PyTorch virtual environment](#load-modules-before-pytorch-virtual-environment)
    - [Load modules before TensorFlow virtual environment](#load-modules-before-tensorflow-virtual-environment)
  - [Test with `pytest`](#test-with-pytest)
- [Working with Docker containers](#working-with-docker-containers)
  - [Terminology Recap](#terminology-recap)
  - [Image Names and Their Purpose](#image-names-and-their-purpose)
  - [Micromamba installation (deprecated)](#micromamba-installation-deprecated)

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
    ml Python CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/11.7
    ml GCCcore/11.3.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 cuDNN
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
    ml Python CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/11.7
    ml GCCcore/11.3.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 cuDNN
    ```

### Install itwinai for users

Install itwinai and its dependencies using the
following command, and follow the instructions:

```bash
# First, load the required environment modules, if on an HPC

# Second, create a python virtual environment and activate it
$ python -m venv ENV_NAME
$ source ENV_NAME/bin/activate

# Install itwinai inside the environment
(ENV_NAME) $ export ML_FRAMEWORK="pytorch" # or "tensorflow"
(ENV_NAME) $ curl -fsSL https://github.com/interTwin-eu/itwinai/raw/main/env-files/itwinai-installer.sh | bash
```

The `ML_FRAMEWORK` environment variable controls whether you are installing
itwinai for PyTorch or TensorFlow.

> [!WARNING]  
> itwinai depends on Horovod, which requires `CMake>=1.13` and
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

You can create the Python virtual environments using our predefined Makefile targets.

#### Installation with pip or uv

You can install the environment using the Cuda wheel, CPU wheel or no wheel at all.
For macOS, you would e.g. use no wheel. This can be done by adding
`--extra-index-url https://download.pytorch.org/whl/<wheel>` at the end of your
`pip install` statement. For example, if you want to download using the cuda wheel
you would do the following:

```bash
pip install -e . --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121
```

If you want to use the CPU wheel instead, you would do the following:

```bash
pip install -e . --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu
```

If you use linux or macOS, make sure to add this to your installation with extras.
For example, if you want to install for macOS as a developer, you would want to add
the `dev` extra and the `macos` extra as follows:

```bash
pip install -e ".[dev,macos]" --no-cache-dir
```

> [!NOTE]
> We use `-e` here because we are in development mode and thus want any changes we make
> to immediately be applied to our venv without having to reinstall.

If you want to use `uv`, which will significantly speed up the installation, you can
prepend `uv` to the `pip` command and it will work in the same way. This assumes that
you already have `uv` installed on your system. An example:

```bash
uv pip install -e . --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121
```

This does not install `Horovod` and `DeepSpeed`, however, as they require a specialized
[script](env-files/torch/install-horovod-deepspeed-cuda.sh). If you do not require
Cuda, then you can install them using `pip` as follows:

```bash
pip install --no-cache-dir --no-build-isolation git+https://github.com/horovod/horovod.git
pip install --no-cache-dir --no-build-isolation deepspeed
```

> [!NOTE]
> It is possible to not use `--no-cache-dir` on a local computer, but on HPC systems we
> recommend using it in order to not fill up your `.cache` directory.

##### Installation using `uv.lock`
The `uv.lock` file provides more information about the exact versions of the libraries
that are used and thus could be better to install from. If you have `uv` installed,
all you need to do is `uv sync` and it will match your `.venv` directory. If you have
installed new packages and wish to update the `uv.lock` file, you can do so with
`uv lock`.

> [!WARNING]
> Even if you have a venv activated while running `uv sync`, the packages will not
> be installed there unless the venv's directory is called exactly `.venv`.

#### PyTorch (+ Lightning) virtual environment

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
    ml Python CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/11.7
    ml GCCcore/11.3.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 cuDNN
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
    ml Python CMake/3.24.3-GCCcore-11.3.0 mpi4py OpenMPI CUDA/11.7
    ml GCCcore/11.3.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0 cuDNN
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
X.Y.Z-[torch|tf]x.y-distro
```

Where:

- `X.Y.Z` is the **itwinai version**
- `x.y` is the **version of the ML framework** (e.g., PyTorch or TensorFlow)
- `distro` is the OS distro in the container (e.g., Ubuntu Jammy)

### Image Names and Their Purpose

We use different image names to group similar images under the same namespace:

- **`itwinai`**: Production images. These should be well-maintained and orderly.
- **`itwinai-dev`**: Development images. Tags can vary, and may include random
hashes.
- **`itwinai-cvmfs`**: Images that need to be made available through CVMFS.

> [!WARNING]
> It is very important to keep the number of tags for `itwinai-cvmfs` as low
> as possible. Tags should only be created under this namespace when strictly
> necessary. Otherwise, this could cause issues for the converter.

<!--
### Micromamba installation (deprecated)

To manage Conda environments we use micromamba, a light weight version of conda.

It is suggested to refer to the
[Manual installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#manual-installation).

Consider that Micromamba can eat a lot of space when building environments because packages are cached on
the local filesystem after being downloaded. To clear cache you can use `micromamba clean -a`.
Micromamba data are kept under the `$HOME` location. However, in some systems, `$HOME` has a limited storage
space and it would be cleverer to install Micromamba in another location with more storage space.
Thus by changing the `$MAMBA_ROOT_PREFIX` variable. See a complete installation example for Linux below, where the
default `$MAMBA_ROOT_PREFIX` is overridden:

```bash
cd $HOME

# Download micromamba (This command is for Linux Intel (x86_64) systems. Find the right one for your system!)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Install micromamba in a custom directory
MAMBA_ROOT_PREFIX='my-mamba-root'
./bin/micromamba shell init $MAMBA_ROOT_PREFIX

# To invoke micromamba from Makefile, you need to add explicitly to $PATH
echo 'PATH="$(dirname $MAMBA_EXE):$PATH"' >> ~/.bashrc
```

**Reference**: [Micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

-->
