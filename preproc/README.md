# Preprocessing module

## Usage

From conda env, as usual. We use local virtual env to
save space in the home directory.

```bash
# Python virtual environment
conda activate -p ./.venv
```

Now you can use our custom CLI:

```bash
itwinpreproc --help
```

<!-- From within a container:

```bash
# Simple container
docker run --rm -it itwin-preproc-light:latest
```

It will open a Python shell. Type the code:

```python
from itwinai import foo

foo.hello()
```

you are executing the code of the Python app
located at `./src`!

Bonus (to be completed):

```bash
# Arch-dependent app
ARCH="linux-64"
docker run --rm -t itwin-ai-$ARCH:latest
``` -->

<!-- ### DevOps

When the virtual environment changes (e.g., new packages are added):

- Export env: this generates/updates environment.yml file
- Generate lock files for reproducibility on different architectures

```bash
make lock
``` -->

## Installation

### Setup development env

Install [Mambaforge](https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux) (suggested),
or any other conda
variant, like miniforge (you can use this [guide](https://abpcomputing.web.cern.ch/guides/python_inst/)).

### Conda / Mamba environment

Below, `mamba` commands can be replaced with `conda` commands.

Install mamba env. To save space in the home directory, we'll create the
environment locally, in the project folder.

```bash
# Install env locally (./.venv folder) and activate it
mamba env create -p ./.venv --file environment.yml
conda activate ./.venv
python -m pip install -e .
```

<!-- ```bash
# Alteratively, install arch-specific env from lock file
mamba create -p ./.venv --file locks/conda-linux-64.lock
mamba activate ./.venv
python -m pip install -e .
``` -->

<!-- ### Build container image

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
``` -->
