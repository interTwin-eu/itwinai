Fast particle detector simulation (CERN) 
========================================

This use case trains a 3D Generative Adversarial Network (3DGAN) for
generation of images of calorimeter depositions. It is based on the
prototype `3DGAN <https://github.com/svalleco/3Dgan/tree/Anglegan/keras>`_ model
developed at CERN and is implemented on PyTorch Lightning framework.

This section covers the CERN use case that utilizes the `torch-lightning` framework 
for training and evaluation. Following you can find instructions to execute CERN use 
case and its integral scripts:

Integration with itwinai
------------------------

.. include:: ../../use-cases/3dgan/README.md
   :parser: myst_parser.sphinx_
   :start-line: 2


3DGAN plugin for itwinai
------------------------

The integration code of the 3DGAN model has been adapted to be distributed as an independent
itwinai plugin called `itwinai-3dgan-plugin <https://github.com/interTwin-eu/itwinai-3dgan-plugin>`_. 


Offloading jobs via interLink
-----------------------------

The CERN use case also has an integration with `interLink <https://github.com/interTwin-eu/interlink>`_. You can find
the relevant files in the 
`interLink directory on Github <https://github.com/interTwin-eu/itwinai/tree/main/use-cases/3dgan/interLink>`_.
You can also look at the README for more information:


.. include:: ../../use-cases/3dgan/interLink/README.md
   :parser: myst_parser.sphinx_
   :start-line: 0
