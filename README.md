# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai.assess.sqaaas/raw/dev/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai.assess.sqaaas/dev/.report/assessment_output.json)

See the latest version of our [docs](https://intertwin-eu.github.io/itwinai/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

## Installation

Requirements:

- Linux environment. Windows and macOS were never tested.

### Micromamba installation

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

### Documentation folder

Documentation for this repository is maintained under `./docs` location.
If you are using code from a previous release, you can build the docs web page
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

#### Test with `pytest`

Do this only if you are a developer wanting to test your code with pytest.

First, you need to create virtual environments both for torch and tensorflow.
For instance, you can use:

```bash
make torch-cpu
make make tf-2.13-cpu
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
# Activate env
micromamba activate ./.venv-pytorch # or ./.venv-tf

pytest -v -m "not slurm" tests/
```

However, some tests are intended to be executed only on an HPC system,
where SLURM is available. They are marked with "slurm" tag. To run also
those tests, use the dedicated job script:

```bash
sbatch tests/slurm_tests_startscript

# Upon completion, check the output:
cat job.err
cat job.out
```
