Pushing and pulling models with the RI-SCALE Model Hub
======================================================
.. include:: ../../../tutorials/model-hub/torch-tutorial-model-hub/README.md
   :parser: myst_parser.sphinx_
   :start-line: 4

Run the training pipeline (trains the model and, as per ``model_hub`` in ``config.yaml``,
pushes the best checkpoint to the Model Hub):

.. code-block:: bash

   itwinai exec-pipeline +pipe-key training_pipeline

Then run the inference pipeline (pulls that same checkpoint and runs inference on it):

.. code-block:: bash

   itwinai exec-pipeline +pipe-key inference_pipeline

config.yaml
+++++++++++
.. literalinclude:: ../../../tutorials/model-hub/torch-tutorial-model-hub/config.yaml
   :language: yaml

data.py
+++++++
.. literalinclude:: ../../../tutorials/model-hub/torch-tutorial-model-hub/data.py
   :language: python

synthetic_data.py
+++++++++++++++++
.. literalinclude:: ../../../tutorials/model-hub/torch-tutorial-model-hub/synthetic_data.py
   :language: python