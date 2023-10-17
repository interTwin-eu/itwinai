# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See the latest version of our [docs](https://intertwin-eu.github.io/T6.5-AI-and-ML/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://intertwin-eu.github.io/T6.5-AI-and-ML/docs/How-to-use-this-software.html).

## Installation

Requirements:

- Linux environment. Windows and macOS were never tested.

### Micromamba installation

To manage Conda environments we use micromamba, a light weight version of conda.

It is suggested to refer to the
[Manual installation guide](https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install).

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

**Reference**: [Micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation.html#micromamba).

### Documentation folder

Documentation for this repository is maintained under `./docs` location.
If you are using code from a previous release, you can build the docs webpage
locally using [these instructions](docs/README#building-and-previewing-your-site-locally).

## Environment setup

Requirements:

- Linux environment. Windows and macOS were never tested.
- Micromamba: see the installation instructions above.
- VS Code, for development.

### TensorFlow

Installation:

```bash
# Install TensorFlow 2.13
make tf-2.13

# Activate env
micromamba activate ./.venv-tf
```

Other TF versions are available, using the following targets `tf-2.10`, and `tf-2.11`.

### PyTorch (+ Lightning)

Installation:

```bash
# Install PyTorch + lightning
make torch-gpu

# Activate env
micromamba activate ./.venv-pytorch
```

Other also CPU-only version is available at the target `torch-cpu`.

### Development environment

This is for developers only. To have it, update the installed `itwinai` package
adding the `dev` extra:

```bash
pip install -e .[dev]
```

To **run tests** on itwinai package:

```bash
# Activate env
micromamba activate ./.venv-pytorch # or ./.venv-tf

pytest -v tests/
```
