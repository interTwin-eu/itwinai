# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See our wiki for a [quick overview](https://github.com/interTwin-eu/T6.5-AI-and-ML/wiki)
of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://github.com/interTwin-eu/T6.5-AI-and-ML/wiki/How-to-use-this-software).

## Installation

Requirements:

- Linux environment
- Mamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html) (suggested Mambaforge).

Install the orchestrator virtual environment.

```bash
# Create local env
make

# Activate env
conda activate ./.venv
```

## Development env setup

Requirements:

- Linux environment
- Mamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html) (suggested Mambaforge).
- VS Code, for development.

Installation:

```bash
make dev-env

# Activate env
conda activate ./.venv-dev
```

## AI environment setup

Requirements:

- Linux environment
- Mamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html) (suggested Mambaforge).

**NOTE**: this environment gets automatically setup when a workflow is executed!

However, you can also set it up explicitly with:

```bash
make ai-env

# Activate env
conda activate ./ai/.venv-pytorch
```
