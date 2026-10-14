Cloud: Kubernetes and Kubeflow
==============================

On Kubernetes you do not write a job script. You describe the job as a manifest and the
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

Examples in this section were run on `NGT <https://ngt.docs.cern.ch>`_, the CERN Kubeflow
platform. Labels such as ``mount-eos`` and the ``default-editor`` service account are specific
to that cluster, so check your own platform's conventions before copying them.

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
           command: ["bash", "-c"]
           args:
             - >-
               exec start-notebook.sh
               --ServerApp.base_url="${NB_PREFIX}"
               --ServerApp.token=""
               --ServerApp.password=""
               --ServerApp.allow_origin="*"
               --ServerApp.ip=0.0.0.0
           resources:
             limits:
               nvidia.com/gpu: 1

Apply it with ``kubectl apply -n <your-namespace> -f notebook.yaml``, then open the notebook
from the Kubeflow dashboard.

.. important::

   The ``command`` override is required. Kubeflow serves notebooks behind a path prefix and
   passes it in ``NB_PREFIX``, but the upstream Jupyter image starts with its own entrypoint
   and ignores that variable, so without the override the notebook starts and every request
   returns 404.

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
