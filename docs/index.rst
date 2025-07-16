.. ============================================================================================
.. Sidenav entries

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🛠️ Installation

   installation/user_installation
   installation/developer_installation
   installation/uv_tutorial

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🚀 Getting started

   getting-started/slurm
   getting-started/containers
   getting-started/plugins
   getting-started/plugins-list
   getting-started/glossary

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🪄 How it works

   how-it-works/training/training
   how-it-works/loggers/explain_loggers
   how-it-works/workflows/explain_workflows
   how-it-works/hpo/explain-hpo
   how-it-works/scalability-report/scalability_report

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🎓 Tutorials

   tutorials/tutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 Scientific Use Cases

   use-cases/use_cases
   use-cases/eurac_doc
   use-cases/virgo_doc
   use-cases/3dgan_doc
   use-cases/cyclones_doc
   use-cases/mnist_doc
   use-cases/xtclim_doc
   use-cases/radio-astronomy
   use-cases/latticeqcd_doc

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: ⚡ API reference

   api/cli_reference
   api/modules

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🎯 Github repository

   itwinai <https://github.com/interTwin-eu/itwinai>

.. ============================================================================================
.. Here the Homepage starts

.. raw:: html

   <div>
     <a href="https://github.com/interTwin-eu/itwinai">
       <img src="https://img.shields.io/github/stars/interTwin-eu/itwinai.svg?style=social&label=Star" alt="GitHub stars" />
     </a>
     &nbsp;
     <a href="https://pypi.org/project/itwinai/">
       <img src="https://img.shields.io/pypi/v/itwinai.svg" alt="PyPI version" />
     </a>
     &nbsp;
     <a href="https://readthedocs.org/projects/itwinai/">
       <img src="https://img.shields.io/readthedocs/itwinai.svg" alt="ReadTheDocs" />
     </a>
   </div>
   </br>

Welcome to **itwinai**
======================

**Accelerate AI & ML workflows** for **Scientific Digital Twins**.

**itwinai** streamlines distributed training, hyperparameter optimization, logging, and modular
workflows, so you can focus on **science**, not plumbing.

Features
--------

- 🚀 **Seamless Scaling**: Run training and inference on HPC clusters or cloud with a single CLI command.
- 🔍 **Effortless Logging**: Built-in support for MLflow, Weights & Biases, TensorBoard, and more.
- 🧩 **Modular Workflows**: Define reusable pipelines for end-to-end experiment management.
- 🤖 **HPO Made Easy**: Native hyperparameter optimization with minimal configuration.
- 🔌 **Extensible Plugins**: Add custom integrations or contribute new features.

Quick Start
-----------

.. code-block:: bash

   # Install via pip
   pip install itwinai

   # Launch a simple distributed training job (assuming a valid config.yaml exists)
   itwinai exec-pipeline

   # View logs in MLflow
   itwinai mlflow-ui --path mllogs/mlflow

🚀 Begin Here
==============

- :doc:`User Installation (for non-developers) <installation/user_installation>`
- :doc:`Developer Installation <installation/developer_installation>`
- :doc:`Submitting jobs to SLURM on HPC <getting-started/slurm>`
- :doc:`Using itwinai Container Images <getting-started/containers>`

🛠️ Core Guides
===============

- :doc:`Training a Neural Network <how-it-works/training/training>`
- :doc:`Logging and Tracking ML workflows <how-it-works/loggers/explain_loggers>`
- :doc:`Defining machine learning workflows <how-it-works/workflows/explain_workflows>`
- :doc:`Hyperparameter Optimization <how-it-works/hpo/explain-hpo>`

🎓 Tutorials
=============

- :doc:`Writing Configuration Files for itwinai <tutorials/workflows/02-pipeline-configuration/tutorial_1_intermediate_workflow>`
- :ref:`Distributed Training <distributed-training-tutorials>`
- :ref:`Hyper-parameter Optimization <hpo-tutorials>`
- :ref:`ML Workflows <ml-workflows-tutorials>`
- :ref:`Code Profiling and Optimization <profiling-tutorials>`

📚 Use Cases & 🧩 Plugins
==========================

- :doc:`MNIST — A Toy Use Case Example <use-cases/mnist_doc>`
- :doc:`Drought Early Warning in the Alps (EURAC) <use-cases/eurac_doc>`
- :doc:`Fast particle detector simulation (CERN) <use-cases/3dgan_doc>`
- :doc:`Writing Plugins for itwinai <getting-started/plugins>`
- :doc:`Current List of itwinai Plugins <getting-started/plugins-list>`

For the full list of scientific use cases refer to the navigation side bar.

⚡ API Reference
================

- :doc:`CLI Reference <api/cli_reference>`
- :doc:`Python SDK <api/modules>`

Integration with EuroHPC centers
================================

Our code has been run and tested on the following EuroHPC systems:

- `LUMI <https://www.lumi-supercomputer.eu/>`_
- `JSC <https://www.fz-juelich.de/jsc/>`_
- `Vega <https://izum.si/en/vega-en/>`_
- `Deucalion <https://eurohpc-ju.europa.eu/supercomputers/our-supercomputers_en#deucalion>`_ (coming soon)

Community & Support
===================

- `GitHub Repository <https://github.com/interTwin-eu/itwinai>`_
- `Contributors <https://github.com/interTwin-eu/itwinai/graphs/contributors>`_
- `interTwin Project <https://www.intertwin.eu/>`_

**itwinai** is an open-source Python library primarily developed by CERN, in collaboration with Forschungszentrum Jülich (FZJ).
As the primary contributor, CERN will retain administrative rights to the repository during and after the interTwin project,
except in cases where CERN is unable to maintain it.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`