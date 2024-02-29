.. itwinai documentation master file, created by
   sphinx-quickstart on Fri Feb  9 13:58:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to itwinai's documentation!
===================================

``itwinai`` is a framework for advanced AI/ML workflows in digital twins.

This platform is intended to support general-purpose MLOps for digital twin use cases in `interTwin <https://www.intertwin.eu/>`_ project.

Platform for machine learning workflows in digital twins
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The goal of this platform is to provide ML researchers with an easy-to-use endpoint to manage general-purpose ML workflows, 
with limited engineering overhead, while providing state-of-the-art MLOps best practices.

The user is going to provide as input a set of configuration files, to fully describe ML workflows, in the context of digital twin applications. 
``itwinai`` platform instantiates ML workflows with the configurations provided by the DT developer.
The execution of ML workflows produces as output a set of ML metrics, which are visualized by ``itwinai`` via `MLFlow <https://mlflow.org/>`_. 
As a result of ML training, the best model (on validation dataset) is saved to the Models Registry for future predictions. 

In ``itwinai`` platform, we focus mainly on the MLOps step, simulating or oversimplifying all the rest (e.g., pre-processing, authentication, workflow execution).


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 💡 Installation

   getting_started_with_itwinai

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🪄 itwinai Modules

   modules

.. toctree::
   :maxdepth: 2
   :caption: 📚 Integrated Use-cases

   use_cases

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🚀 Tutorials

   tutorials

   
**`interTwin Demo: itwinai integration with other DTE modules <https://www.youtube.com/watch?v=NoVCfSxwtX0>`_**
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`

