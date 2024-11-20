How to run a use case
======================

Each use case comes with their own tutorial on how to run it. Before running them,
however, you should set up a Python virtual environment. Refer to the
:doc:`getting started section <../getting-started/getting_started_with_itwinai.rst`
for more information on how to do this.

After installing and activating the virtual environment, you will want to install the
use-case specific dependencies, if applicable. This can be done by first ``cd``-ing
into the use-case directory and then installing the requirements, as follows

.. code-block:: bash

   cd use-cases/<name-of-use-case>
   pip install -r requirements.txt


Alternatively, you can use the use-case Docker image, if available. After setting
everything up, you can now run the use case as specified in the use case's tutorial.


Fast particle detector simulation | CERN 
=================================================

The first ``interTwin`` use case integrated with ``itwinai`` framework is the DT for
fast particle detector simulation. 3D Generative Adversarial Network (3DGAN) for
generation of images of calorimeter depositions. This project is based on the
prototype `3DGAN <https://github.com/svalleco/3Dgan/tree/Anglegan/keras>`_ model
developed at CERN and is implemented on PyTorch Lightning framework.

.. toctree::
   :maxdepth: 2

   3dgan_doc


MNIST dataset 
=========================

MNIST image classification is used to provide an example on how to define an end-to-end
digital twin workflow with the ``itwinai`` software.

.. toctree::
   :maxdepth: 2

   mnist_doc


Tropical Cyclones Detection | CMCC 
==============================================

You can find more information on the ``itwinai`` integration of the Tropical Cyclones
(TCs) Detection model, developed by CMCC, in the
:doc:`Tropical Cyclones Detection documentation <cyclones_doc>`.


Noise Simulation for Gravitational Waves Detector (Virgo) | INFN 
===========================================================================

You can find more information on the Virgo use-case integration with the ``itwinai``
framework in the :doc:`Virgo documentation <virgo_doc>`.


Drought Early Warning in the Alps | EURAC 
===========================================

You can find more information on the EURAC use-case integration with the ``itwinai``
in the :doc:`EURAC documentation <eurac_doc>`.
