Cloud: RayJob
=============

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
       rayVersion: '2.54.0'
       headGroupSpec:
         rayStartParams:
           num-cpus: '0'
         template:
           spec:
             serviceAccountName: default-editor
             containers:
             - name: ray-head
               image: ghcr.io/intertwin-eu/itwinai:0.4.2-slim-torch2.6-noble
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
               image: ghcr.io/intertwin-eu/itwinai:0.4.2-slim-torch2.6-noble
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

Ray Tune prints one row per trial at the end:

.. code-block:: text

   Trial status: 4 TERMINATED
   Logical resource usage: 5.0/16 CPUs, 1.0/2 GPUs (0.0/2.0 accelerator_type:L40S)
   +--------------------------+------------+------------+-------------+------+---------+
   | Trial name               | status     | batch_size | optim_lr    | iter | loss    |
   +--------------------------+------------+------------+-------------+------+---------+
   | TorchTrainer_7e03c_00000 | TERMINATED |         64 | 0.000458146 |    2 | 0.48263 |
   | TorchTrainer_7e03c_00001 | TERMINATED |        128 | 0.000215905 |    2 | 1.96529 |
   | TorchTrainer_7e03c_00002 | TERMINATED |         64 | 0.000262630 |    2 | 1.96492 |
   | TorchTrainer_7e03c_00003 | TERMINATED |         64 | 0.000281781 |    2 | 0.47591 |
   +--------------------------+------------+------------+-------------+------+---------+

Trading GPUs for trials is the usual tuning decision: ``ray_num_workers`` spends GPUs on making
one trial faster, ``ray_num_trials`` spends them on exploring more configurations.

Gotchas
-------

.. warning::

   **A RayJob can report ``SUCCEEDED`` while your training failed.** The job status reflects
   the entrypoint process, not the Ray Tune trials. A trial that raises still leaves
   ``jobStatus: SUCCEEDED``. Always read the trial table: ``TERMINATED`` means the trial
   finished, ``ERROR`` means it did not.

Other things worth knowing:

- **The service account is not optional.** Ray workers talk to the Kubernetes API to find the
  head, so a Kubeflow cluster needs ``serviceAccountName: default-editor`` on head and workers.
  Without it the cluster never becomes ready.
- **Give the head no GPUs and no CPUs for scheduling.** ``num-cpus: '0'`` on the head keeps Ray
  from placing training workers there.
- **Pin the NCCL interface.** On a pod network, ``NCCL_SOCKET_IFNAME=eth0`` and
  ``NCCL_IB_DISABLE=1`` avoid a hang while NCCL probes interfaces it cannot use.
- **The first run on a fresh node is slow.** The itwinai image is large, so expect a few
  minutes of image pull before anything starts. Subsequent jobs on the same node start at once.
- **Your cluster may restrict GPU counts and models.** Admission policies often allow only
  certain GPU counts per pod, and a ``nodeSelector`` on the GPU model is frequently required.
  Check your platform's documentation.

What to read next
-----------------

- :doc:`cloud-pipelines` to chain this into a multi-step pipeline, or to run DDP without Ray.
- :doc:`../how-it-works/hpo/explain-hpo` for schedulers, metrics and search algorithms.
