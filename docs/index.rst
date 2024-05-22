.. itwinai documentation master file, created by
   sphinx-quickstart on Fri Feb  9 13:58:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🚧 UNDER CONSTRUCTION 🚧
=========================

Welcome to itwinai's documentation!
===================================

``itwinai`` is a framework for advanced AI/ML workflows in Digital Twins (DTs).

This platform is intended to support general-purpose MLOps for Digital Twin use cases in the `interTwin <https://www.intertwin.eu/>`_ project.

Platform for machine learning workflows in digital twins
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The goal of this platform is to provide ML researchers with an easy-to-use endpoint to manage general-purpose ML workflows, 
with limited engineering overhead, while providing state-of-the-art MLOps best practices.

The user can fully describe ML workflows for DT applications by providing a set of configuration files as input.
The ``itwinai`` platform instantiates ML workflows with the configurations provided by the DT developer.
The execution of ML workflows outputs a set of ML metrix, which are visualised by ``itwinai`` via `MLFlow <https://mlflow.org/>`_.
The trained ML model that performed best on the validation dataset is saved to the Models Registry for future predictions.

In ``itwinai`` platform, we focus mainly on the MLOps step, simulating or oversimplifying the rest (e.g., pre-processing, authentication, workflow execution).


.. toctree::
   :numbered:
   :maxdepth: 2
   :hidden:
   :caption: 💡 Installation

   getting_started_with_itwinai

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 Integrated Use Cases

   use_cases

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🚀 Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🪄 Python API reference

   modules

   
`interTwin Demo: itwinai integration with other DTE modules <https://www.youtube.com/watch?v=NoVCfSxwtX0>`_
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. * :ref:`search`

