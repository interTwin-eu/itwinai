# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

The entry point of a workflow is given by the orchestrator script `run-workflow.py` .

See some examples of workflow executions in `examples.sh` , for instance:

```bash
conda run -p ./.venv python run-workflow.py -f ./use-cases/mnist/training-workflow.yml
```

## Installation

Requirements:

- Mamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html) (suggested Mambaforge).

Install the orchestrator virtual environment.

```bash
mamba env create -p ./.venv --file environment.yml
conda activate ./.venv
```
