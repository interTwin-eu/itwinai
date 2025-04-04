# itwinai

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai.assess.sqaaas/main/.report/assessment_output.json)

![itwinai Logo](./docs/images/icon-itwinai-orange-black-subtitle.png)

`itwinai` is a Python toolkit designed to help scientists and researchers streamline AI and machine learning
workflows, specifically for digital twin applications. It provides easy-to-use tools for distributed training,
hyper-parameter optimization on HPC systems, and integrated ML logging, reducing engineering overhead and accelerating
research. Developed primarily by CERN, in collaboration with Forschungszentrum Jülich (FZJ), `itwinai` supports modular
and reusable ML workflows, with the flexibility to be extended through third-party plugins, empowering AI-driven scientific
research in digital twins.

See the latest version of our docs [here](https://itwinai.readthedocs.io/).

> [!WARNING]
> Branch protection rules are applied to all branches which names
> match this regex: `[dm][ea][vi]*` . When creating new branches,
> please avoid using names that match that regex, otherwise branch
> protection rules will block direct pushes to that branch.

## Installation

For instructions on how to install `itwinai`, please refer to the
[user installation guide](https://itwinai.readthedocs.io/latest/installation/user_installation.html)
or the
[developer installation guide](https://itwinai.readthedocs.io/latest/installation/developer_installation.html),
depending on whether you are a user or developer

For information about how to use containers or how to test with pytest, you can look
at the following documents:

- [Working with containers](/docs/working-with-containers.md)
- [Testing with pytest](/docs/testing-with-pytest.md)
