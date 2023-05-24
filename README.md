# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See the latest version of our [docs](https://intertwin-eu.github.io/T6.5-AI-and-ML/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://intertwin-eu.github.io/T6.5-AI-and-ML/docs/How-to-use-this-software.html).

## Installation

Requirements:

- Linux environment
- Micromamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html#micromamba).

Install the orchestrator virtual environment.

```bash
# Create local env
make

# Activate env
conda activate ./.venv
```

To run tests on workflows use:

```bash
# Activate env
conda activate ./.venv

pytest tests/
```

### Documentation folder

Documentation for this repository is maintained under `./docs` location.
If you are using code from a previous release, you can build the docs webpage
locally using [these instructions](docs/README#building-and-previewing-your-site-locally).

## Development env setup

Requirements:

- Linux environment
- Micromamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html#micromamba).
- VS Code, for development.

Installation:

```bash
make dev-env

# Activate env
conda activate ./.venv-dev
```

To run tests on itwinai package:

```bash
# Activate env
conda activate ./.venv-dev

pytest tests/ai/
```

To lock conda env files for ai workflows, after they have been updated:

```bash
conda activate ./.venv

make lock-ai
```

## AI environment setup

Requirements:

- Linux environment
- Micromamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html#micromamba).

**NOTE**: this environment gets automatically setup when a workflow is executed!

However, you can also set it up explicitly with:

```bash
make ai-env

# Activate env
conda activate ./ai/.venv-pytorch
```
