Notebook
========

A Kubeflow ``Notebook`` gives you a JupyterLab with a GPU, your code and the shared
filesystem, which is the quickest way to try itwinai out.

The easiest way is the **Kubeflow dashboard**: Notebooks, then New Notebook, then paste
``ghcr.io/intertwin-eu/itwinai:jlab-slim-latest`` as the custom image and pick how many GPUs
you want. The image handles the Kubeflow contract by itself, so there is nothing else to set.
See :doc:`../getting-started/containers` for the tag scheme and for building your own image
on top of it.

Inside the notebook, itwinai is already installed and the CLI works as anywhere else:

.. code-block:: bash

   itwinai exec-pipeline --config-name config.yaml +pipe_key=training_pipeline epochs=2

