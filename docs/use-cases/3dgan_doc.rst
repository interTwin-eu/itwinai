3DGAN
=====

This section covers the CERN use case that utilizes the `torch-lightning` framework for training and evaluation. Following you can find instructions to execute CERN use case and its integral scripts:

itwinai x 3DGAN
---------------

.. include:: ../../use-cases/3dgan/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2

.. toctree::
   :maxdepth: 5

model.py
++++++++

.. literalinclude:: ../../use-cases/3dgan/model.py
   :language: python


trainer.py
++++++++++
.. literalinclude:: ../../use-cases/3dgan/trainer.py
   :language: python


saver.py
++++++++

.. literalinclude:: ../../use-cases/3dgan/saver.py
   :language: python


dataloader.py
+++++++++++++

.. literalinclude:: ../../use-cases/3dgan/dataloader.py
   :language: python


config.yaml
+++++++++++

This YAML file defines the pipeline configuration for the CERN use case.

.. literalinclude:: ../../use-cases/3dgan/config.yaml
   :language: yaml


create_inference_sample.py
++++++++++++++++++++++++++

This file defines a pipeline configuration for the CERN use case inference.

.. literalinclude:: ../../use-cases/3dgan/create_inference_sample.py
   :language: python


Dockerfile
++++++++++

.. literalinclude:: ../../use-cases/3dgan/Dockerfile
   :language: bash


startscript
+++++++++++

.. literalinclude:: ../../use-cases/3dgan/startscript
   :language: bash



This section covers the CERN use case integration with `interLink <https://github.com/interTwin-eu/interLink>`_ using ``itwinai``. The following files are integral to this use case:

interLink x 3DGAN
-----------------

.. toctree::
   :maxdepth: 5


3dgan-inference-cpu.yaml
++++++++++++++++++++++++

.. literalinclude:: ../../use-cases/3dgan/interLink/3dgan-inference-cpu.yaml
   :language: yaml


3dgan-inference.yaml
++++++++++++++++++++

.. literalinclude:: ../../use-cases/3dgan/interLink/3dgan-inference.yaml
   :language: yaml


3dgan-train.yaml
++++++++++++++++

.. literalinclude:: ../../use-cases/3dgan/interLink/3dgan-train.yaml
   :language: yaml



.. .. automodule:: 3dgan.model
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. .. automodule:: 3dgan.train
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. .. automodule:: 3dgan.trainer
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. .. automodule:: 3dgan.saver
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. .. automodule:: 3dgan.dataloader
..     :members:
..     :undoc-members:
..     :show-inheritance:
