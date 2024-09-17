# Read The Docs documentation page

The python dependencies are organized in two requirements files, which
must be installed in the following order:

1. `pre-requirements.txt` contains torch and tensorflow.
1. `requirements.txt` contains the packages which depend on torch and tensorflow,
which should be installed *after* torch and tensorflow.

## Build docs locally

To build the docs locally and visualize them in your browser, without relying on external
services (e.g., Read The Docs cloud), use the following commands

```bash
# Clone the repo, if not done yet
git clone --recurse-submodules https://github.com/interTwin-eu/itwinai.git itwinai-docs
cd itwinai-docs

# The first time, you may need to install some Linux packages (assuming Ubuntu system here)
sudo apt update && sudo apt install libmysqlclient-dev
sudo apt install python3-sphinx

# Create a python virtual environment and install itwinai and its dependencies
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r docs/pre-requirements.txt
pip install -r docs/requirements.txt
pip install sphinx-rtd-theme

# Move to the docs folder and build them using Sphinx
cd docs
make clean
make html

# Serve a local HTTP server to navigate the newly created docs pages.
# You can see the docs visiting http://localhost:8000 in your browser.
python -m http.server --directory  _build/html/
```

### Build docs on JSC

On JSC systems, the way of building the docs locally is similar to the method
explained above. However, the environment setup must be slightly adapted to use
some modules provided on the HPC system.

To manage the docs, you can simply use the Makefile target
belows.

From the repository's root, create the docs virtual environment:

```bash
make docs-env-jsc
```

Once the environment is ready, build the docs
and serve them on localhost:

```bash
make docs-jsc
```

## Read The Docs management page

To manage the documentation on Read The Docs (RTD) cloud, visit
[https://readthedocs.org/projects/itwinai](https://readthedocs.org/projects/itwinai/).
