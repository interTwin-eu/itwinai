RayJob
======

A ``RayJob`` creates a short-lived Ray cluster, runs one command on it, and deletes it. This is
the recommended way to do multi-GPU and multi-node training on Kubernetes with itwinai,
because the same manifest covers both distributed training and hyperparameter optimization:
only the overrides change.

It needs the `KubeRay operator <https://docs.ray.io/en/latest/cluster/kubernetes/index.html>`_
on the cluster. See :doc:`cloud` for the other prerequisites.

Why Ray
-------

itwinai detects a running Ray cluster and swaps the distributed strategy for its Ray
equivalent, so ``strategy=ddp`` becomes :class:`~itwinai.torch.distributed.RayDDPStrategy`
without any change to your configuration. Ray then places one worker per GPU, across nodes if
needed, and it is the same mechanism that drives HPO. See
:doc:`../how-it-works/hpo/explain-hpo` for the concepts.

Distributed training
--------------------

Two GPUs, one per worker. Raise ``replicas`` to scale out: Ray does not care whether the
workers land on one node or several.

.. code-block:: yaml
   :caption: ``rayjob-train.yaml``

   apiVersion: ray.io/v1
   kind: RayJob
   metadata:
     name: itwinai-mnist-ddp
   spec:
     shutdownAfterJobFinishes: true
     ttlSecondsAfterFinished: 600
     entrypoint: >-
       cd /shared/<user>/mnist-usecase &&
       itwinai exec-pipeline --config-name config.yaml
       +pipe_key=training_pipeline
       strategy=ddp
       epochs=2
       dataset_root=/shared/<user>/mnist-usecase/data
       ray_num_workers=2
       ray_gpus_per_worker=1
       ray_cpus_per_worker=3
     rayClusterSpec:
       rayVersion: '2.54.0'  # must match the image, see below
       headGroupSpec:
         rayStartParams:
           num-cpus: '0'
         template:
           spec:
             serviceAccountName: default-editor
             containers:
             - name: ray-head
               image: ghcr.io/intertwin-eu/itwinai:torch-slim-latest
               env:
               - name: PYTHONPATH
                 value: /shared/<user>/mnist-usecase
               - name: NCCL_SOCKET_IFNAME
                 value: eth0
               - name: NCCL_IB_DISABLE
                 value: "1"
               resources:
                 limits:   {cpu: "2", memory: "8G"}
                 requests: {cpu: "500m", memory: "4G"}
       workerGroupSpecs:
       - replicas: 2
         groupName: gpu-group
         rayStartParams:
           num-cpus: '8'
           num-gpus: '1'
         template:
           spec:
             serviceAccountName: default-editor
             nodeSelector:
               nvidia.com/gpu.product: NVIDIA-L40S
             containers:
             - name: ray-worker
               image: ghcr.io/intertwin-eu/itwinai:torch-slim-latest
               env:
               - name: PYTHONPATH
                 value: /shared/<user>/mnist-usecase
               - name: NCCL_SOCKET_IFNAME
                 value: eth0
               - name: NCCL_IB_DISABLE
                 value: "1"
               resources:
                 limits:
                   nvidia.com/gpu: 1

``rayVersion`` has to match the Ray version in the image. itwinai does not pin Ray, so the
version moves with the image tag: read it with ``docker run --rm <image> ray --version`` and
update the manifest when you change tag.

Submit it with ``kubectl apply -n <your-namespace> -f rayjob-train.yaml``, and follow it with
``kubectl logs -n <your-namespace> -f job/itwinai-mnist-ddp``.

The three ``ray_*`` overrides are the only knobs you normally touch:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Override
     - Meaning
   * - ``ray_num_workers``
     - Total training processes. Must equal the total number of GPUs you gave the worker group.
   * - ``ray_gpus_per_worker``
     - GPUs per process. Keep it at 1 unless you are sharing a GPU between workers.
   * - ``ray_cpus_per_worker``
     - CPUs per process, for the data loaders. Must fit in the worker's ``num-cpus``.

Hyperparameter optimization
---------------------------

Same manifest, same config file. Ask for more than one trial and itwinai hands the search to
Ray Tune:

.. code-block:: yaml
   :caption: entrypoint for HPO

   entrypoint: >-
     cd /shared/<user>/mnist-usecase &&
     itwinai exec-pipeline --config-name config.yaml
     +pipe_key=training_pipeline
     strategy=ddp
     epochs=2
     dataset_root=/shared/<user>/mnist-usecase/data
     ray_num_workers=1
     ray_gpus_per_worker=1
     ray_cpus_per_worker=4
     ray_num_trials=4

With two GPUs available and one GPU per trial, two trials run at a time and the rest queue.
The search space is part of the pipeline configuration, not the manifest:

.. code-block:: yaml
   :caption: ``config.yaml``, inside the trainer

   ray_search_space:
     batch_size:
       type: choice
       categories: [64, 128]
     optim_lr:
       type: uniform
       lower: 1e-5
       upper: 1e-3

What to read next
-----------------

- :doc:`cloud-pipelines` to chain this into a multi-step pipeline, or
  :doc:`cloud-pytorchjob` to run DDP without Ray.
- :doc:`../how-it-works/hpo/explain-hpo` for schedulers, metrics and search algorithms.
