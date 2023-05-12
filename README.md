# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

The entry point of a workflow is given by the orchestrator script `run-workflow.py` .

See some examples of workflow executions in `examples.sh` , for instance:

```bash
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml
```
If you want to run using the Common Workflow Language (CWL). Note that logging and saving models/metric is currently not supported using CWL.

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
