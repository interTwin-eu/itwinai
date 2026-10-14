Cloud
=====

On Kubernetes in a cloud environment, you do not write a job script. You describe the job as a manifest and the
cluster schedules it. itwinai ships container images, so there is nothing to install on the
nodes.

Every example on these pages was run on the CERN IT Kubeflow Platform.

There are four ways to run, and they answer different questions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use
     - When
   * - :doc:`Notebook <cloud-notebook>`
     - Exploring, debugging, a single GPU. Start here.
   * - :doc:`RayJob <cloud-ray>`
     - Distributed training and HPO, multi-GPU and multi-node.
   * - :doc:`PyTorchJob <cloud-pytorchjob>`
     - Distributed training without Ray, using ``torchrun`` directly.
   * - :doc:`Kubeflow pipeline <cloud-pipelines>`
     - Chaining steps: prepare data, train, evaluate, with parameters and a run history.

They compose: a pipeline step can create a ``RayJob`` or a ``PyTorchJob`` instead of
training in its own pod, which keeps the orchestration in Kubeflow and the scaling in Ray.

What the cluster must provide
-----------------------------

- **GPU nodes**, with the NVIDIA device plugin.
- **A shared filesystem** mounted in every pod, for the dataset, the checkpoints and the
  logs. All examples below use ``/shared``.
- **The KubeRay operator** for the RayJob path, and **Kubeflow** for notebooks, pipelines and
  PyTorchJob.

Where to put data and code
--------------------------

Put both on the shared filesystem and point the job at them:

.. code-block:: text

   /shared/<user>/my-usecase/
   |-- config.yaml
   |-- dataloader.py
   |-- model.py
   `-- data/

itwinai images do not contain your use case, so set ``PYTHONPATH`` to that directory for
Hydra to import the classes named by ``_target_``:

.. code-block:: bash

   cd /shared/<user>/my-usecase
   export PYTHONPATH=/shared/<user>/my-usecase
   itwinai exec-pipeline --config-name config.yaml +pipe_key=training_pipeline

For anything more permanent, build your use case into an image or an
:doc:`itwinai plugin <../getting-started/plugins>`.

.. toctree::
   :maxdepth: 1

   cloud-notebook
   cloud-ray
   cloud-pipelines
   cloud-pytorchjob
