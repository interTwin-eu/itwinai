MNIST
=====

This section covers the MNIST use case, which utilizes the `torch-lightning` framework for training and evaluation. The following files are integral to this use case:

Torch Lightning
---------------

**Training**

.. code-block:: bash

   # Download dataset and exit: only run first step in the pipeline (index=0)
   itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 0

   # Run the whole training pipeline
   itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline 


View training logs on MLFLow server (if activated from the configuration):

.. code-block:: bash

   mlflow ui --backend-store-uri mllogs/mlflow/


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

config.yaml
+++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../use-cases/mnist/torch-lightning/config.yaml
   :language: yaml

startscript
+++++++++++

The `startscript` is a shell script to initiate the training process. It sets up the environment and starts the training using the `train.py` script.

.. literalinclude:: ../use-cases/mnist/torch-lightning/startscript
   :language: bash


utils.py
++++++++

The `utils.py` script includes utility functions and classes that are used across the MNIST use case.

.. literalinclude:: ../use-cases/mnist/torch-lightning/utils.py
   :language: python



This section covers the MNIST use case, which utilizes the `torch` framework for training and evaluation. The following files are integral to this use case:

PyTorch
-------

**Training**

.. code-block:: bash

   # Download dataset and exit
   itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps dataloading_step

   # Run the whole training pipeline
   itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline 


View training logs on MLFLow server (if activated from the configuration):

.. code-block:: bash

   mlflow ui --backend-store-uri mllogs/mlflow/


**Inference**

1. Create sample dataset

   .. code-block:: python

      from dataloader import InferenceMNIST
      InferenceMNIST.generate_jpg_sample('mnist-sample-data/', 10)
    

2. Generate a dummy pre-trained neural network

   .. code-block:: python

      import torch
      from model import Net
      dummy_nn = Net()
      torch.save(dummy_nn, 'mnist-pre-trained.pth')
   

3. Run inference command. This will generate a "mnist-predictions" folder containing a CSV file with the predictions as rows.

   .. code-block:: bash
      
      itwinai exec-pipeline --config config.yaml --pipe-key inference_pipeline 
    

Note the same entry point as for training.

Docker image
++++++++++++

Build from project root with

.. code-block:: bash

   # Local
   docker buildx build -t itwinai:0.0.1-mnist-torch-0.1 -f use-cases/mnist/torch/Dockerfile .

   # Ghcr.io
   docker buildx build -t ghcr.io/intertwin-eu/itwinai:0.0.1-mnist-torch-0.1 -f use-cases/mnist/torch/Dockerfile .
   docker push ghcr.io/intertwin-eu/itwinai:0.0.1-mnist-torch-0.1


**Training with Docker container**

.. code-block:: bash

   docker run -it --rm --name running-inference \
      -v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai:0.01-mnist-torch-0.1 \
      /bin/bash -c "itwinai exec-pipeline --print-config \
      --config /usr/src/app/config.yaml \
      --pipe-key training_pipeline \
      -o dataset_root=/usr/data/mnist-dataset "


**Inference with Docker container**

From wherever a sample of MNIST jpg images is available
(folder called 'mnist-sample-data/'):

::

   ├── $PWD
   │   ├── mnist-sample-data
   │   │   ├── digit_0.jpg
   │   │   ├── digit_1.jpg
   │   │   ├── digit_2.jpg
   ...
   │   │   ├── digit_N.jpg


.. code-block:: bash
   
   docker run -it --rm --name running-inference \
      -v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai:0.01-mnist-torch-0.1 \
      /bin/bash -c "itwinai exec-pipeline --print-config \
      --config /usr/src/app/config.yaml \
      --pipe-key inference_pipeline \
      -o test_data_path=/usr/data/mnist-sample-data \
      -o inference_model_mlflow_uri=/usr/src/app/mnist-pre-trained.pth \
      -o predictions_dir=/usr/data/mnist-predictions "


This command will store the results in a folder called "mnist-predictions":

::

   ├── $PWD
   │   ├── mnist-predictions
   |   │   ├── predictions.csv



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


create_inference_sample.py
++++++++++++++++++++++++++

This file defines a pipeline configuration for the MNIST use case inference.

.. literalinclude:: ../use-cases/mnist/torch/create_inference_sample.py
   :language: python

model.py
++++++++

The `model.py` script is responsible for loading a simple model.

.. literalinclude:: ../use-cases/mnist/torch/model.py
   :language: python

config.yaml
+++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../use-cases/mnist/torch/config.yaml
   :language: yaml

startscript.sh
++++++++++++++

The `startscript` is a shell script to initiate the training process. It sets up the environment and starts the training using the `train.py` script.

.. literalinclude:: ../use-cases/mnist/torch/startscript.sh
   :language: bash


saver.py
++++++++
...

.. literalinclude:: ../use-cases/mnist/torch/saver.py
   :language: python


runall.sh
+++++++++

.. literalinclude:: ../use-cases/mnist/torch/runall.sh
   :language: bash


slurm.sh
++++++++

.. literalinclude:: ../use-cases/mnist/torch/slurm.sh
   :language: bash



This section covers the MNIST use case, which utilizes the `tensorflow` framework for training and evaluation. The following files are integral to this use case:

Tensorflow
----------

.. toctree::
   :maxdepth: 5

dataloader.py
+++++++++++++

The `dataloader.py` script is responsible for loading the MNIST dataset and preparing it for training.

.. literalinclude:: ../use-cases/mnist/tensorflow/dataloader.py
   :language: python


pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the MNIST use case. It includes settings for the model, training, and evaluation.

.. literalinclude:: ../use-cases/mnist/tensorflow/pipeline.yaml
   :language: yaml


startscript.sh
++++++++++++++

The `startscript` is a shell script to initiate the training pipeline.

.. literalinclude:: ../use-cases/mnist/tensorflow/startscript.sh
   :language: bash



   