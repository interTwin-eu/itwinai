Tropical Cyclones Detection
===========================

The code is adapted from the CMCC use case's
`repository <https://github.com/CMCC-Foundation/ml-tropical-cyclones-detection>`_.

To know more on the interTwin tropical cyclones detection use case and its DT, please visit the published deliverables,
`D4.1 <https://zenodo.org/records/10417135>`_, 
`D7.1 <https://zenodo.org/records/10417158>`_ and `D7.3 <https://zenodo.org/records/10224252>`_.

.. include:: ../../use-cases/cyclones/README.md
   :parser: myst_parser.sphinx_
   :start-line: 5

pipeline.yaml
+++++++++++++

This YAML file defines the pipeline configuration for the CMCC use case.

.. literalinclude:: ../../use-cases/cyclones/pipeline.yaml
   :language: yaml

train.py
++++++++++
.. literalinclude:: ../../use-cases/cyclones/train.py
   :language: python

dataloader.py
+++++++++++++

.. literalinclude:: ../../use-cases/cyclones/dataloader.py
   :language: python

trainer.py
++++++++++
.. literalinclude:: ../../use-cases/cyclones/trainer.py
   :language: python

startscript
+++++++++++

.. literalinclude:: ../../use-cases/cyclones/startscript.sh
   :language: bash

cyclones_vgg.py
+++++++++++++++

.. literalinclude:: ../../use-cases/cyclones/cyclones_vgg.py
   :language: python
