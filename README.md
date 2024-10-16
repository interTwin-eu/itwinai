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

You can create the
Python virtual environments using our predefined Makefile targets.

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
