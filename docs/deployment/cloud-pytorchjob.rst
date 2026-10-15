PyTorchJob
==========

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
               image: ghcr.io/intertwin-eu/itwinai:torch-slim-latest
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

   Wrap the command in ``torchrun``. The training operator injects ``RANK``,
   ``WORLD_SIZE``, ``MASTER_ADDR`` and ``MASTER_PORT``, but not ``LOCAL_RANK`` and
   ``LOCAL_WORLD_SIZE``, which itwinai needs to recognise a distributed environment. Running
   ``itwinai exec-pipeline`` directly would silently train on one GPU per pod, with each pod
   unaware of the others. ``torchrun`` fills in the missing variables, and the operator's
   values feed it through ``--nnodes`` and ``--node_rank``.

