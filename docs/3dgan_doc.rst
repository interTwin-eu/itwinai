3DGAN
=====

This section covers the CERN use case, which utilizes the `torch-lightning` framework for training and evaluation. The following files are integral to this use case:

itwinai x 3DGAN
---------------


.. toctree::
   :maxdepth: 5


model.py
++++++++

.. literalinclude:: ../use-cases/3dgan/model.py
   :language: python


trainer.py
++++++++++
.. literalinclude:: ../use-cases/3dgan/trainer.py
   :language: python


saver.py
++++++++

.. literalinclude:: ../use-cases/3dgan/saver.py
   :language: python


dataloader.py
+++++++++++++

.. literalinclude:: ../use-cases/3dgan/dataloader.py
   :language: python


cern-pipeline.yaml
++++++++++++++++++

This YAML file defines the pipeline configuration for the CERN use case.

.. literalinclude:: ../use-cases/3dgan/cern-pipeline.yaml
   :language: yaml


inference-pipeline.yaml
+++++++++++++++++++++++

This YAML file defines the pipeline configuration for the CERN use case inference.

.. literalinclude:: ../use-cases/3dgan/inference-pipeline.yaml
   :language: yaml


Dockerfile
++++++++++

.. literalinclude:: ../use-cases/3dgan/Dockerfile
   :language: bash


pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the CERN use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../use-cases/3dgan/pipeline.yaml
   :language: yaml



This section covers the CERN use case integration with `interLink <https://github.com/interTwin-eu/interLink>`_ using ``itwinai``. The following files are integral to this use case:

interLink x 3DGAN
-----------------

.. toctree::
   :maxdepth: 5


3dgan-inference-cpu.yaml
++++++++++++++++++++++++

.. literalinclude:: ../use-cases/3dgan/interLink/3dgan-inference-cpu.yaml
   :language: yaml


3dgan-inference.yaml
++++++++++++++++++++++++

.. literalinclude:: ../use-cases/3dgan/interLink/3dgan-inference.yaml
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
