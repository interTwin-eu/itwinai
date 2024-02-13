---
layout: default
title: CLI
nav_order: 3
---

# Command-line interface (CLI)
<!-- {: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
-->

The `itwinai` package provides a custom CLI, which can be accessed, for instance
from the development environment:

```bash
# Activate development environment
micromamba activate ./.venv-dev

# Access itwinai CLI
itwinai --help
```

## Visualization

Some visualization functionalities offered by `itwinai` CLI.

```bash
# Datasets registry
itwinai datasets --help

# Workflows (any file '*-workflow.yml')
itwinai workflows --help
```

## Machine learning

```bash
# Training
itwinai train --help

# Launch MLFlow UI to visualize ML logs
itwinai mlflow-ui --help

# Inference
itwinai predict --help
```
