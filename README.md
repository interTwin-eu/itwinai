# interTwin T6.5

Got to the [wiki](https://gitlab.cern.ch/mbunino/intertwin-t6.5/-/wikis/home) to check the current progress.

## Usage

From conda env, as usual.

```bash
# Python virtual environment
conda activate -n itwin-ai
```

From within a container:

```bash
# Simple container
docker run -it itwin-ai-light:latest
```

It will open a Python shell. Type the code:

```python
from testai import foo

foo.hello()
```

you are executing the code of the Python app
located at `./src`!

Bonus (to be completed):

```bash
# Arch-dependent app
ARCH="linux-64"
docker run -t itwin-ai-$ARCH:latest
```

### DevOps

When the virtual environment changes (e.g., new packages are added):

- Export env: this generates/updates environment.yml file
- Generate lock files for reproducibility on different architectures

```bash
make lock
```

## Installation

### Setup development env

Install [Mambaforge](https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux) (suggested),
or any other conda
variant, like miniforge (you can use this [guide](https://abpcomputing.web.cern.ch/guides/python_inst/)).

### Conda / Mamba environment

Below, `mamba` command can be replaced with `conda` command.

Install mamba env for local usage

```bash
# Install generic env and activate it
mamba env create -n itwin-ai --file environment.yml
mamba activate itwin-ai

# Alteratively, install arch-specific env from lock file
mamba create -n itwin-ai --file locks/conda-linux-64.lock
mamba activate itwin-ai
```

### Build container image

For fast prototyping, you can build a simple container with:

```bash
make light
```

When building the container image for some specific platform (i.e., architecture),
you need to specify the architecture on which you are planning to deploy the container.
Some examples:

- `linux-64`: Linux for Intel x64 arch
- `linux-aarch64`: Linux for ARM64 arch
- `linux-ppc64le`

```bash
ARCH="linux-64"
make $ARCH
```
