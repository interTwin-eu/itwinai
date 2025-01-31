# Read The Docs documentation page

The docs can be built either locally on your system or remotely on JSC.

## Build docs locally

To build the docs locally and visualize them in your browser without relying on external
services (e.g., Read The Docs cloud), follow these steps:

### Step 0 - Clone the Repository

If you haven't already cloned the repository, you can do it like this:

```bash
git clone --recurse-submodules https://github.com/interTwin-eu/itwinai.git itwinai-docs
cd itwinai-docs
```

Notice the `--recurse-submodules` flag here, which makes sure that any `git submodules`
are also installed.

### Step 1 - Install Linux Packages

You might need to install some Linux packages. With an Ubuntu system, you can use the
following commands:

```bash
sudo apt update && sudo apt install libmysqlclient-dev
sudo apt install pandoc
sudo apt install python3-sphinx
```

### Step 2 - Create a Virtual Environment and Install itwinai

We first build a virtual environment and then install `itwinai` with the `docs` and
`torch` extras. If you didn't clone the repository recursively, then you also have to
update submodules. All of this is done with the following commands:

```bash
# Update submodules
git submodule update --init --recursive

# Create venv
python3 -m venv .venv-docs
source .venv-docs/bin/activate

# Install itwinai
pip install ".[torch,docs]"
```

### Step 3 - Build the docs and start a server

Now you can go into the right directory, clean any old build files and then build
the docs. Finally, you can start a server to look at the docs. This can all be done
as follows:

```bash
# Move to the docs folder and build them using Sphinx
cd docs
make clean
make html

# Serve a local HTTP server to navigate the newly created docs pages.
# You can see the docs by visiting http://localhost:8000 in your browser.
python -m http.server --directory  _build/html/
```

### Build docs on JSC

On JSC systems, the way of building the docs locally is similar to the method
explained above. However, the environment setup must be slightly adapted to use
some modules provided on the HPC system.

To manage the docs, you can simply use the Makefile target
below.

From the repository's root, create the docs virtual environment:

```bash
make docs-env-jsc
```

Once the environment is ready, build the docs and serve them on localhost:

```bash
make docs-jsc
```

## Read The Docs management page

To manage the documentation on Read The Docs (RTD) cloud, visit
[https://readthedocs.org/projects/itwinai](https://readthedocs.org/projects/itwinai/).
