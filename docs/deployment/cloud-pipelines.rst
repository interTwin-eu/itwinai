Cloud: Kubeflow pipelines and PyTorchJob
========================================

Two Kubeflow paths that are not Ray. Use a **pipeline** when training is one step among
several and you want parameters and a run history. Use a **PyTorchJob** when you want plain
``torchrun`` across pods, without a Ray cluster.

Kubeflow pipelines
------------------

A Kubeflow pipeline is a graph of containers. itwinai fits as one container per step, and
because a pipeline step is just a command, you reuse the same ``config.yaml`` and the same
``itwinai exec-pipeline`` call as everywhere else.

A useful split is a cheap CPU step that prepares the dataset, followed by a GPU step that
trains. ``+pipe_steps`` selects part of the itwinai pipeline, so you do not need two config
files:

.. code-block:: python
   :caption: ``mnist_pipeline.py``

   # pip install "kfp==2.13.0" "kfp-kubernetes==1.5.0"
   from kfp import dsl, kubernetes

   IMAGE = "ghcr.io/intertwin-eu/itwinai:0.4.2-slim-torch2.6-noble"
   WORKDIR = "/shared/<user>/mnist-usecase"
   GPU_MODEL = "NVIDIA-L40S"
   GPUS = 2


   @dsl.container_component
   def prepare_data(workdir: str):
       """Run only the data-getter step of the itwinai pipeline."""
       return dsl.ContainerSpec(
           image=IMAGE,
           command=["bash", "-c"],
           args=[
               f"cd {workdir} && export PYTHONPATH={workdir} && "
               "itwinai exec-pipeline --config-name config.yaml "
               "+pipe_key=training_pipeline +pipe_steps=[dataloading_step] "
               f"dataset_root={workdir}/data"
           ],
       )


   @dsl.container_component
   def train(workdir: str, epochs: int):
       """Run the full itwinai pipeline under torchrun, single node, GPUS processes."""
       return dsl.ContainerSpec(
           image=IMAGE,
           command=["bash", "-c"],
           args=[
               f"cd {workdir} && export PYTHONPATH={workdir} && "
               f"torchrun --nnodes=1 --nproc_per_node={GPUS} --no-python "
               "itwinai exec-pipeline --config-name config.yaml "
               "+pipe_key=training_pipeline strategy=ddp "
               f"epochs={epochs} dataset_root={workdir}/data "
               f"checkpoints_location={workdir}/checkpoints_kfp"
           ],
       )


   @dsl.pipeline(name="itwinai-mnist")
   def mnist_pipeline(epochs: int = 2):
       prep = prepare_data(workdir=WORKDIR)
       prep.set_caching_options(False)

       tr = train(workdir=WORKDIR, epochs=epochs)
       tr.after(prep)
       tr.set_caching_options(False)
       tr.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(GPUS)
       kubernetes.add_node_selector(tr, "nvidia.com/gpu.product", GPU_MODEL)

Compile and submit it:

.. code-block:: python

   from kfp import compiler
   from kfp.client import Client

   compiler.Compiler().compile(mnist_pipeline, "mnist_pipeline.yaml")

   client = Client()  # inside a notebook on the cluster
   client.create_run_from_pipeline_func(
       mnist_pipeline,
       namespace="<your-namespace>",
       experiment_name="itwinai-smoke",
       arguments={"epochs": 2},
   )

From a notebook on the cluster, ``Client()`` finds the pipeline service by itself. From your
laptop you need a token:

.. code-block:: bash

   TOKEN=$(kubectl create token default-editor \
     --duration=2h \
     --audience=istio-ingressgateway.istio-system.svc.cluster.local)

.. code-block:: python

   client = Client(host="https://<your-kubeflow-host>/pipeline", existing_token=TOKEN)

You can also upload ``mnist_pipeline.yaml`` through the Kubeflow dashboard and start runs from
there.

.. warning::

   **Pass a literal to ``set_accelerator_limit()``.** A pipeline parameter compiles to a
   placeholder that is never resolved into the pod's resource request, so the step silently
   starts with no GPU, and on a cluster with tainted GPU nodes it then stays ``Pending``
   forever. Keep the GPU count a Python constant, as ``GPUS`` is above.

Two smaller ones:

- ``+pipe_steps`` takes step **names** when the pipeline uses named steps, as the MNIST
  configuration does. Indices are rejected.
- Caching is on by default and will skip a step whose inputs have not changed. That is
  usually right, and confusing the first time you rerun a job expecting it to retrain.

PyTorchJob
----------

A ``PyTorchJob`` runs one pod per node and wires the rendezvous for you. Use it when you want
DDP, DeepSpeed or Horovod across pods without a Ray cluster.

.. code-block:: yaml
   :caption: ``pytorchjob.yaml``

   apiVersion: kubeflow.org/v1
   kind: PyTorchJob
   metadata:
     name: itwinai-mnist-torchrun
   spec:
     runPolicy:
       ttlSecondsAfterFinished: 1800
     pytorchReplicaSpecs:
       Master:
         replicas: 1
         restartPolicy: Never
         template: &tpl
           spec:
             serviceAccountName: default-editor
             nodeSelector:
               nvidia.com/gpu.product: NVIDIA-L40S
             containers:
             - name: pytorch
               image: ghcr.io/intertwin-eu/itwinai:0.4.2-slim-torch2.6-noble
               command: ["bash", "-c"]
               args:
                 - >-
                   cd /shared/<user>/mnist-usecase &&
                   export PYTHONPATH=/shared/<user>/mnist-usecase &&
                   torchrun
                   --nnodes="$WORLD_SIZE"
                   --nproc_per_node=1
                   --node_rank="$RANK"
                   --master_addr="$MASTER_ADDR"
                   --master_port="$MASTER_PORT"
                   --no-python
                   itwinai exec-pipeline --config-name config.yaml
                   +pipe_key=training_pipeline
                   strategy=ddp
                   epochs=2
                   dataset_root=/shared/<user>/mnist-usecase/data
               env:
               - name: NCCL_SOCKET_IFNAME
                 value: eth0
               - name: NCCL_IB_DISABLE
                 value: "1"
               resources:
                 limits:
                   nvidia.com/gpu: 1
       Worker:
         replicas: 1
         restartPolicy: Never
         template: *tpl

Two replicas with one GPU each gives a world size of 2. For more nodes, raise ``Worker``
replicas; for more GPUs per node, raise both ``nvidia.com/gpu`` and ``--nproc_per_node``.

.. important::

   **Wrap the command in ``torchrun``.** The training operator injects ``RANK``,
   ``WORLD_SIZE``, ``MASTER_ADDR`` and ``MASTER_PORT``, but not ``LOCAL_RANK`` and
   ``LOCAL_WORLD_SIZE``, which itwinai needs to recognise a distributed environment. Running
   ``itwinai exec-pipeline`` directly would silently train on one GPU per pod, with each pod
   unaware of the others. ``torchrun`` fills in the missing variables, and the operator's
   values feed it through ``--nnodes`` and ``--node_rank``.

To confirm that a run is genuinely distributed, set ``ITWINAI_LOG_LEVEL=DEBUG`` and look for
one line per pod:

.. code-block:: text

   [DEBUG] itwinai.distributed:224 - Detected torchrun/TorchElastic distributed environment
   [DEBUG] itwinai.torch.trainer:332 - Enough resources? True (distributed_resources_available: True, ray_cluster_is_running: False)

Choosing between the three
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Path
     - Pick it when
   * - :doc:`RayJob <cloud-ray>`
     - Default for distributed training, and the only path for HPO.
   * - PyTorchJob
     - You want ``torchrun`` semantics, or you are already standardised on the training
       operator.
   * - Pipeline
     - Training is one step of several, or you want parameterized, repeatable runs with a UI.

They compose: a pipeline step can create a ``RayJob`` or a ``PyTorchJob`` instead of training
in its own pod, which keeps the orchestration in Kubeflow and the scaling in Ray.
