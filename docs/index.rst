Welcome to itwinai's documentation!
===================================

``itwinai`` is a versatile toolkit designed to accelerate AI and machine learning (ML) workflows for researchers and scientists, 
particularly in the realm of Digital Twins (DTs). This toolkit provides a suite of user-friendly tools to effortlessly
scale machine learning projects to high-performance computing (HPC) resources, seamlessly integrating with cloud-based services.
The primary focus of ``itwinai`` is to reduce the engineering burden on researchers, enabling them to concentrate more on advancing
their science.

Empowering AI in Scientific Digital Twins
+++++++++++++++++++++++++++++++++++++++++

The ``itwinai`` toolkit is engineered to support AI-driven research in scientific digital twins. It offers powerful capabilities for
distributed machine learning training and inference on HPC systems, efficient hyper-parameter optimization (HPO), and simplified
ML logging with integration to popular tools like MLflow, Weights & Biases, and TensorBoard. Additionally, it includes an
intuitive framework to define, configure, and manage modular and reusable ML workflows, providing a streamlined approach to
experiment management.

Moreover, the toolkit is designed with extensibility in mind, allowing third-party developers to build and integrate their own
plugins, enhancing the flexibility and adaptability of the platform.

``itwinai`` is primarily developed by `Matteo Bunino <matteo.bunino@cern.ch>`_ from `CERN <https://home.cern/>`_, in
collaboration with the `interTwin <https://www.intertwin.eu/>`_
project, which aims to advance the use of digital twins in scientific research.


How to Read the Docs
++++++++++++++++++++

To effectively utilize the ``itwinai`` toolkit documentation, begin by exploring the "Getting Started" section. This part is essential
for grasping the basics and setting up the toolkit, with detailed instructions for different installation scenarios, whether on HPC
systems or your local machine.

For a deeper dive into the core functionalities, check out the "How It Works" section, which breaks down the key concepts that power
``itwinai``. The "Scientific Use Cases" section offers practical examples and scenarios from the `interTwin <https://www.intertwin.eu/>`_
project, showcasing how the toolkit is applied in real-world research.

Enhance your skills by exploring the "Tutorials" section, filled with step-by-step guides on distributed ML training and workflow
creation. Lastly, the "Python API Reference" is your go-to resource for a detailed overview of the toolkit's capabilities, helping
you implement specific features in your projects.

Following these sections systematically will help you maximize your understanding and make the most of the ``itwinai`` toolkit in
your research endeavors.

``itwinai`` documentation is also available in different versions: 'latest', 'stable', and specific release versions like 'v0.2.1'.
The 'latest' version reflects the most recent updates, while the 'stable' version is recommended for production use, as it
contains thoroughly tested features aligned with the toolkit's most recent release
(`learn more <https://docs.readthedocs.io/en/stable/versions.html#version-states>`_).


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 💡 Getting started

   getting-started/getting_started_with_itwinai
   getting-started/slurm
   getting-started/plugins

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🪄 How it works

   how-it-works/training/training
   how-it-works/loggers/explain_loggers
   how-it-works/workflows/explain_workflows
   how-it-works/hpo/explain-hpo

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🚀 Tutorials

   tutorials/tutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 Scientific Use Cases

   use-cases/use_cases

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: ⚡ Python API reference

   python-api/modules

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🎯 Github repository

   itwinai <https://github.com/interTwin-eu/itwinai>


.. .. toctree::
..    :maxdepth: 2
..    :hidden:
..    :caption: Additional resources

..    notebooks/example


interTwin Demo: itwinai integration with other DTE modules 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. raw:: html
   
   <iframe width="560" height="315" src="https://www.youtube.com/embed/NoVCfSxwtX0" title="interTwin demo: itwinai (WP6)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


|
|
|


Indices and tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`

.. * :ref:`search`

