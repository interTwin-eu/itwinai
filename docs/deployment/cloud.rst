Cloud: Kubernetes and Kubeflow
==============================

On Kubernetes in a cloud environment, you do not write a job script. You describe the job as a manifest and the
cluster schedules it. itwinai ships container images, so there is nothing to install on the
nodes.

There are four ways to run, and they answer different questions:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Use
     - When
   * - Jupyter notebook
     - Exploring, debugging, a single GPU. Start here.
   * - :doc:`RayJob <cloud-ray>`
     - Distributed training and HPO, multi-GPU and multi-node.
   * - :doc:`PyTorchJob <cloud-pipelines>`
     - Distributed training without Ray, using ``torchrun`` directly.
   * - :doc:`Kubeflow pipeline <cloud-pipelines>`
     - Chaining steps: prepare data, train, evaluate, with parameters and a run history.

What the cluster must provide
-----------------------------

- **GPU nodes**, with the NVIDIA device plugin.
- **A shared filesystem** mounted in every pod, for the dataset, the checkpoints and the
  logs. All examples below use ``/shared``.
- **The KubeRay operator** for the RayJob path, and **Kubeflow** for notebooks, pipelines and
  PyTorchJob.

Examples in this section were run on the CERN Kubeflow platform. Labels such as
``mount-eos`` and the ``default-editor`` service account are specific to that
cluster, so check your own platform's conventions before copying them.

Images
------

.. code-block:: text

   ghcr.io/intertwin-eu/itwinai:0.4.2-slim-torch2.6-noble        # training
   ghcr.io/intertwin-eu/itwinai:jlab-slim-0.4.2-torch2.6-noble   # JupyterLab

.. note::

   Use the images from ``ghcr.io/intertwin-eu`` on Kubernetes. The ``registry.cern.ch/itwinai``
   registry holds Singularity images, which are for :doc:`HPC <hpc>`.

See :doc:`../getting-started/containers` for the tag scheme and for building your own image on
top of these.

Interactive: a Jupyter notebook
-------------------------------

A Kubeflow ``Notebook`` gives you a JupyterLab with a GPU, your code and the shared
filesystem, which is the quickest way to try itwinai out.

The easiest way is the **Kubeflow dashboard**: Notebooks, then New Notebook, then paste the
JupyterLab image as a custom image and pick how many GPUs you want. The image handles the
Kubeflow contract by itself, so there is nothing else to set.

For something repeatable, the same notebook as a manifest:

.. code-block:: yaml
   :caption: ``notebook.yaml``

   apiVersion: kubeflow.org/v1
   kind: Notebook
   metadata:
     name: itwinai-lab
     labels:
       app: itwinai-lab
       home-eos: "true"
       mount-eos: "true"
       mount-cvmfs: "true"
   spec:
     template:
       spec:
         serviceAccountName: default-editor
         containers:
         - name: itwinai-lab
           image: ghcr.io/intertwin-eu/itwinai:jlab-slim-0.4.2-torch2.6-noble
           resources:
             limits:
               nvidia.com/gpu: 1

Apply it with ``kubectl apply -n <your-namespace> -f notebook.yaml``, then open the notebook
from the Kubeflow dashboard. The container name has to match the notebook name.

.. note::

   **What the image does for Kubeflow, and what to do with older images.** Kubeflow serves
   each notebook behind a path prefix, passes it in ``NB_PREFIX``, and runs the container as
   the logged-in user's UID with a ``/etc/passwd`` of its own. The entrypoint handles all
   three: it serves JupyterLab under ``NB_PREFIX``, it skips the upstream Jupyter start
   script when that script's ``jovyan`` user does not exist, and it starts even when the
   optional Rucio configuration cannot be written.

   Images up to ``jlab-slim-0.4.2`` do none of that, so started from the dashboard they fail
   with ``PermissionError: '/opt/rucio/etc/rucio.cfg'``. With one of those, override the
   command in a manifest instead:

   .. code-block:: yaml

      command: ["bash", "-c"]
      args:
        - >-
          exec start-notebook.sh
          --ServerApp.base_url="${NB_PREFIX}"
          --ServerApp.token=""
          --ServerApp.password=""
          --ServerApp.allow_origin="*"
          --ServerApp.ip=0.0.0.0

Inside the notebook, itwinai is already installed and the CLI works as anywhere else:

.. code-block:: bash

   itwinai exec-pipeline --config-name config.yaml +pipe_key=training_pipeline epochs=2

One GPU is enough to check that a pipeline runs. To use more than one, move to a
:doc:`RayJob <cloud-ray>`.

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
