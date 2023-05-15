# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See our wiki for a [quick overview](https://github.com/interTwin-eu/T6.5-AI-and-ML/wiki)
of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://github.com/interTwin-eu/T6.5-AI-and-ML/wiki/How-to-use-this-software).

If you want to run using the Common Workflow Language (CWL).
Note that logging and saving models/metric is currently not supported using CWL.

```bash
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow-cwl.yml --cwl
```

## Installation

Requirements:

- Linux environment
- Mamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html) (suggested Mambaforge).
- VS Code, for development.

Install the orchestrator virtual environment.

```bash
mamba env create -p ./.venv --file environment-cern.yml
conda activate ./.venv
```
