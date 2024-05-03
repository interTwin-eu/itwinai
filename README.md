# itwinai

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai.assess.sqaaas/raw/dev/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai.assess.sqaaas/dev/.report/assessment_output.json)

See the latest version of our [docs](https://intertwin-eu.github.io/itwinai/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

## Installation

Requirements:

- Linux environment. Windows and macOS were never tested.

### Python virtual environment

Depending on your environment, there are different ways to
select a specific python version.

#### Laptop or GPU node

If you are working on a laptop
or on a simple on-prem setup, you could consider using
[pyenv](https://github.com/pyenv/pyenv). See the
[installation instructions](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation). If you are using pyenv,
make sure to read [this](https://github.com/pyenv/pyenv/wiki#suggested-build-environment).

#### Install itwinai environment

Regardless of how you loaded your environment, you can create the
python virtual environments with the following commands.
Once the correct Python version is loaded, create the virtual
environments using our pre-make Makefile:

```bash
make torch-env # or make torch-env-cpu
make tensorflow-env # or make tensorflow-env-cpu

# Juelich supercomputer
make torch-gpu-jsc
make tf-gpu-jsc
```

## Environment setup

Requirements:

- Linux environment. Windows and macOS were never tested.
- VS Code, for development.

### TensorFlow

Installation:

```bash
# Install TensorFlow 2.13
make tensorflow-env

# Activate env
source .venv-tf/bin/activate
```

A CPU-only version is available at the target `tensorflow-env-cpu`.

### PyTorch (+ Lightning)

Installation:

```bash
# Install PyTorch + lightning
make torch-env

# Activate env
source .venv-pytorch/bin/activate
```

A CPU-only version is available at the target `torch-env-cpu`.

### Development environment

This is for developers only. To have it, update the installed `itwinai` package
adding the `dev` extra:

```bash
pip install -e .[dev]
```

#### Test with `pytest`

Do this only if you are a developer wanting to test your code with pytest.

First, you need to create virtual environments both for torch and tensorflow.
For instance, you can use:

```bash
make torch-env-cpu
make tensorflow-env-cpu
```

To select the name of the torch and tf environments you can set the following
environment variables, which allow to run the tests in environments with
custom names which are different from `.venv-pytorch` and `.venv-tf`.

```bash
export TORCH_ENV="my_torch_env"
export TF_ENV="my_tf_env"
```

Functional tests (marked with `pytest.mark.functional`) will be executed under
`/tmp/pytest` location to guarantee they are run in a clean environment.

To run functional tests use:

```bash
pytest -v tests/ -m "functional"
```

To run all tests on itwinai package:

```bash
make test
```

Run tests in JSC virtual environments:

```bash
make test-jsc
```

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
