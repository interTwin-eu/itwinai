# PoC for AI-centric digital twin workflows

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

## Running a workflow in CWL

The entry point of a workflow is given by the orchestrator script `workflow.cwl` . You can run it with any tool that supports the CWL worklfow format, e.g. `cwltool`.

See some examples of workflow executions in `examples.sh` , for instance:

```bash
cwltool workflow.cwl use-cases/mnist/training-workflow-cwl.yml
```

To modify the workflow edit the `use-cases/mnist/training-workflow-cwl.yml` file.
You can visualize the workflow and store it in a vector graphic:

```bash
cwltool --print-dot workflow.cwl | dot -Tsvg > workflow_graph.svg
```

## Installation

Requirements:

- Mamba: [Installation guide](https://mamba.readthedocs.io/en/latest/installation.html) (suggested Mambaforge).

To install the `cwltool` and activate the enviroment to run the workflow:

```
mamba create --name cwltool_test cwltool -c conda-forge
conda activate cwltool
```

You need to install the virtual environement with the following commands for preprocessing and training:

```
mamba env create -p ./use-cases/mnist/.venv-preproc --file ./use-cases/mnist/preproc-env.yml
```

If you want training with GPU support with CUDA run:

```
mamba env create -p ./ai/.venv-training-gpu --file ./ai/training-env-gpu.yml
```

otherwise, to install PyTorch with only CPU support:

```
mamba env create -p ./ai/.venv-training --file ./ai/training-env.yml
```

To install the `itwinai` module run:

```
conda run -p ./ai/.venv-training-gpu python -m pip install --no-deps ./ai
```