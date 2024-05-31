Integrated Use Cases
====================

Here you can find a collection of use cases showing how ``itwinai`` can be used. Each use case folder contains:

- ``pipeline.yaml``: textual description of the ML workflow for that use case
- ``train.py``: entry point of training workflow.
- ``startscript``: file to execute the training workflow on a SLURM-based cluster.
- ``requirements.txt``: (optional) use case-specific requirements. can be installed with:
  
.. code-block:: bash

  cd use/case/folder
  # After activating the correct environment...
  pip install -r requirements.txt


How to run a use case
---------------------

First, create the use case's Python environment (i.e., PyTorch or TensorFlow)
as described `here <https://itwinai.readthedocs.io/latest/getting_started_with_itwinai.html#environment-setup>`_, and activate it.
Then, install use case-specific dependencies, if any:

.. code-block:: bash

   pip install -r /use/case/path/requirements.txt


Alternatively, you can use the use case Docker image, if available.

Then, go to the use case's directory:

.. code-block:: bash

   cd /use/case/path


From here you can run the use case (having activated the correct Python env):

.. code-block:: bash

   # Locally
   python train.py [OPTIONS...]

   # With SLURM: stdout and stderr will be saved to job.out and job.err files
   sbatch startscript



Fast particle detector simulation | CERN use case
-------------------------------------------------

The first ``interTwin`` use case integrated with ``itwinai`` framework is the DT for fast particle detector simulation. 
3D Generative Adversarial Network (3DGAN) for generation of images of calorimeter depositions. 
This project is based on the prototype `3DGAN <https://github.com/svalleco/3Dgan/tree/Anglegan/keras>`_ model developed at CERN and is implemented on PyTorch Lightning framework.

.. toctree::
   :maxdepth: 2

   3dgan_doc


MNIST dataset use case
----------------------

MNIST image classification is used to provide an example on 
how to define an end-to-end digital twin workflow with the ``itwinai`` software.

.. toctree::
   :maxdepth: 2

   mnist_doc


Tropical Cyclones Detection | CMCC use case
-------------------------------------------

Below you can find the training and validation of a Tropical Cyclones (TCs) Detection model, developed by CMCC, integrated with ``itwinai`` framework.

.. toctree::
   :maxdepth: 1

   cyclones_doc



Noise Simulation for Gravitational Waves Detector (Virgo) | INFN use case
-------------------------------------------------------------------------

Below you can find the integration of the Virgo use case with ``itwinai`` framework, developed by INFN.

.. toctree::
   :maxdepth: 1

   virgo_doc

