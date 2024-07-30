MNIST
=====

This section covers the MNIST use case, which utilizes the `torch-lightning` framework for training and evaluation. The following files are integral to this use case:

Torch Lightning
---------------

.. include:: ../../use-cases/mnist/torch-lightning/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2


.. toctree::
   :maxdepth: 5

dataloader.py
+++++++++++++

The `dataloader.py` script is responsible for loading the MNIST dataset and preparing it for training.

.. literalinclude:: ../../use-cases/mnist/torch-lightning/dataloader.py
   :language: python

.. .. automodule:: torch-lightning.dataloader
..    :members:
..    :undoc-members:
..    :show-inheritance:

config.yaml
+++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../../use-cases/mnist/torch-lightning/config.yaml
   :language: yaml

startscript
+++++++++++

The `startscript` is a shell script to initiate the training process. It sets up the environment and starts the training using the `train.py` script.

.. literalinclude:: ../../use-cases/mnist/torch-lightning/startscript
   :language: bash


utils.py
++++++++

The `utils.py` script includes utility functions and classes that are used across the MNIST use case.

.. literalinclude:: ../../use-cases/mnist/torch-lightning/utils.py
   :language: python



This section covers the MNIST use case, which utilizes the `torch` framework for training and evaluation. The following files are integral to this use case:

PyTorch
-------

.. include:: ../../use-cases/mnist/torch/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2



.. toctree::
   :maxdepth: 5

dataloader.py
+++++++++++++

The `dataloader.py` script is responsible for loading the MNIST dataset and preparing it for training.

.. literalinclude:: ../../use-cases/mnist/torch/dataloader.py
   :language: python


Dockerfile
++++++++++

.. literalinclude:: ../../use-cases/mnist/torch/Dockerfile
   :language: bash


create_inference_sample.py
++++++++++++++++++++++++++

This file defines a pipeline configuration for the MNIST use case inference.

.. literalinclude:: ../../use-cases/mnist/torch/create_inference_sample.py
   :language: python

model.py
++++++++

The `model.py` script is responsible for loading a simple model.

.. literalinclude:: ../../use-cases/mnist/torch/model.py
   :language: python

config.yaml
+++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../../use-cases/mnist/torch/config.yaml
   :language: yaml

startscript.sh
++++++++++++++

The `startscript` is a shell script to initiate the training process. It sets up the environment and starts the training using the `train.py` script.

.. literalinclude:: ../../use-cases/mnist/torch/startscript.sh
   :language: bash


saver.py
++++++++
...

.. literalinclude:: ../../use-cases/mnist/torch/saver.py
   :language: python


runall.sh
+++++++++

.. literalinclude:: ../../use-cases/mnist/torch/runall.sh
   :language: bash


slurm.sh
++++++++

.. literalinclude:: ../../use-cases/mnist/torch/slurm.sh
   :language: bash



This section covers the MNIST use case, which utilizes the `tensorflow` framework for training and evaluation. The following files are integral to this use case:

Tensorflow
----------

.. toctree::
   :maxdepth: 5

dataloader.py
+++++++++++++

The `dataloader.py` script is responsible for loading the MNIST dataset and preparing it for training.

.. literalinclude:: ../../use-cases/mnist/tensorflow/dataloader.py
   :language: python


pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../../use-cases/mnist/tensorflow/pipeline.yaml
   :language: yaml


startscript.sh
++++++++++++++

The `startscript` is a shell script to initiate the training pipeline.

.. literalinclude:: ../../use-cases/mnist/tensorflow/startscript.sh
   :language: bash



   