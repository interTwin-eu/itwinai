Picking a mode
==============

itwinai runs the same way everywhere. Your pipeline configuration, your trainer and the
``itwinai exec-pipeline`` command do not change when you move from a laptop to a
supercomputer or to a Kubernetes cluster in the cloud. What changes is the **launcher**: the
thing that allocates GPUs and starts one process per GPU.

This section covers the launchers.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Mode
     - Compute
     - You launch with
   * - :doc:`HPC <hpc>`
     - Supercomputer, SLURM queue
     - ``sbatch``, or ``itwinai generate-slurm``
   * - :doc:`Cloud <cloud>`
     - Kubernetes GPU nodes
     - A Kubeflow Notebook, a ``RayJob``, a ``PyTorchJob`` or a Kubeflow pipeline
   * - :doc:`Cloud to HPC <cloud-hpc>`
     - Supercomputer, driven from Kubernetes
     - Kubernetes manifests, offloaded by interLink

What is identical in every mode
-------------------------------

- **The configuration.** One ``config.yaml`` describes the pipeline. The same file was used
  for every example in this section.
- **The command.** ``itwinai exec-pipeline`` with Hydra-style overrides.
- **The strategy choice.** ``strategy=ddp`` (or ``deepspeed``, ``horovod``) asks for
  distributed training. itwinai then picks the concrete implementation from the environment it
  finds: a Ray cluster gives you :class:`~itwinai.torch.distributed.RayDDPStrategy`, a
  ``torchrun`` rendezvous gives you :class:`~itwinai.torch.distributed.TorchDDPStrategy`, and
  a single process gives you :class:`~itwinai.torch.distributed.NonDistributedStrategy`. You
  do not edit the config to switch.

So porting a use case between modes is a change of job script, not a change of code.

Which one
---------

- You have an HPC allocation and a queue: use :doc:`HPC <hpc>`.
- You have a Kubernetes cluster with GPUs: use :doc:`Cloud <cloud>`. Start with a notebook to
  explore, then submit a ``RayJob`` to scale out.
- You have both, and the data or the GPUs are on the HPC side while your orchestration lives
  in Kubernetes: use :doc:`Cloud to HPC <cloud-hpc>`.
