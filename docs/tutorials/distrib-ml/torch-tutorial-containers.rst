itwinai and containers (Docker and Singularity)
===================================================

In this tutorial you will learn how to use itwinai's containers images to run your ML workflows
without having to setup the python environment by means of virtual environments.

.. include:: ../../../tutorials/distributed-ml/torch-tutorial-containers/README.md
   :parser: myst_parser.sphinx_


Shell scripts
--------------

run_docker.sh
++++++++++++++++
.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-containers/run_docker.sh
   :language: bash

slurm.sh
++++++++++++
.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-containers/slurm.sh
   :language: bash


runall.sh
++++++++++++++++
.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-containers/runall.sh
   :language: bash


Pipeline configuration
-----------------------

config.yaml
++++++++++++

.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-containers/config.yaml
   :language: yaml


Python files 
------------------

model.py
++++++++++++

.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-containers/model.py
   :language: python

dataloader.py
+++++++++++++++
.. literalinclude:: ../../../tutorials/distributed-ml/torch-tutorial-containers/dataloader.py
   :language: python


