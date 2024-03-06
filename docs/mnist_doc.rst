MNIST
=====

This section covers the MNIST use case, which utilizes the `torch-lightning` framework for training and evaluation. The following files are integral to this use case:

torch-lightning
---------------

.. toctree::
   :maxdepth: 5

dataloader.py
+++++++++++++

The `dataloader.py` script is responsible for loading the MNIST dataset and preparing it for training.

.. literalinclude:: ../use-cases/mnist/torch-lightning/dataloader.py
   :language: python

.. .. automodule:: torch-lightning.dataloader
..    :members:
..    :undoc-members:
..    :show-inheritance:

pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../use-cases/mnist/torch-lightning/pipeline.yaml
   :language: yaml

startscript
+++++++++++

The `startscript` is a shell script to initiate the training process. It sets up the environment and starts the training using the `train.py` script.

.. literalinclude:: ../use-cases/mnist/torch-lightning/startscript
   :language: bash

train.py
++++++++

This script contains the training loop and is where the model is trained using the data prepared by `dataloader.py`.

.. literalinclude:: ../use-cases/mnist/torch-lightning/train.py
   :language: python

.. .. automodule:: torch-lightning.train
..    :members:
..    :undoc-members:
..    :show-inheritance:

trainer.py
++++++++++

The `trainer.py` file defines the `Trainer` class which sets up the training parameters and the training process.

.. literalinclude:: ../use-cases/mnist/torch-lightning/trainer.py
   :language: python

.. .. automodule:: torch-lightning.trainer
..    :members:
..    :undoc-members:
..    :show-inheritance:

utils.py
++++++++

The `utils.py` script includes utility functions and classes that are used across the MNIST use case.

.. literalinclude:: ../use-cases/mnist/torch-lightning/utils.py
   :language: python

.. .. automodule:: torch-lightning.utils
..    :members:
..    :undoc-members:
..    :show-inheritance:


This section covers the MNIST use case, which utilizes the `torch` framework for training and evaluation. The following files are integral to this use case:

torch
-----

.. toctree::
   :maxdepth: 5

dataloader.py
+++++++++++++

The `dataloader.py` script is responsible for loading the MNIST dataset and preparing it for training.

.. literalinclude:: ../use-cases/mnist/torch/dataloader.py
   :language: python


Dockerfile
++++++++++

.. literalinclude:: ../use-cases/mnist/torch/Dockerfile
   :language: bash


inference-pipeline.yaml
+++++++++++++++++++++++

This YAML file defines the pipeline configuration for the MNIST use case inference.

.. literalinclude:: ../use-cases/mnist/torch/inference-pipeline.yaml
   :language: yaml

model.py
++++++++

The `model.py` script is responsible for loading a simple model.

.. literalinclude:: ../use-cases/mnist/torch/model.py
   :language: python

pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../use-cases/mnist/torch/pipeline.yaml
   :language: yaml

startscript
+++++++++++

The `startscript` is a shell script to initiate the training process. It sets up the environment and starts the training using the `train.py` script.

.. literalinclude:: ../use-cases/mnist/torch/startscript
   :language: bash

train.py
++++++++

This script contains the training loop and is where the model is trained using the data prepared by `dataloader.py`.

.. literalinclude:: ../use-cases/mnist/torch/train.py
   :language: python

saver.py
++++++++
...

.. literalinclude:: ../use-cases/mnist/torch/saver.py
   :language: python

