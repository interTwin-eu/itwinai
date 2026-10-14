Cloud to HPC: interLink
=======================

Sometimes the orchestration belongs in Kubernetes but the compute belongs on HPC: the GPUs are
there, or the dataset is too large to move, or the allocation is already granted.
`interLink <https://github.com/interTwin-eu/interLink>`_ makes that possible without running
two separate workflows.

How it works
------------

interLink adds a `virtual kubelet <https://virtual-kubelet.io>`_ to your Kubernetes cluster: a
node that reports itself as schedulable but owns no hardware. When a pod is scheduled onto it,
interLink translates the pod into a batch job on the remote HPC system, submits it through the
site's SLURM, and streams status and logs back. From the Kubernetes side it is an ordinary pod.
On the HPC side it is an ordinary SLURM job running a Singularity container.

What this buys you:

- **One control plane.** Your pipeline, your RayJob and your monitoring stay in Kubernetes.
- **No SSH in your workflow.** You do not script ``sbatch`` over SSH, and you do not need an
  interactive session to launch anything.
- **HPC-grade interconnect** where it matters, since the training itself runs on the HPC nodes.

What it costs you:

- **Latency.** A pod becomes a queued SLURM job, so startup is at the mercy of the queue.
- **A site agreement.** interLink needs a component installed on, or reachable from, the HPC
  site, which is a conversation with the site administrators rather than a local decision.
- **Images must be staged.** The container has to be on the HPC filesystem as a Singularity
  ``.sif`` beforehand. See :doc:`hpc`.

Requesting HPC resources
------------------------

SLURM options travel as pod annotations. interLink reads them and builds the job script:

.. code-block:: yaml

   metadata:
     annotations:
       slurm-job.vk.io/flags: >-
         -p gpu --gres=gpu:4 --gpus-per-node=4
         --ntasks-per-node=1 --nodes=2 --cpus-per-task=32
         --time=01:00:00
       slurm-job.vk.io/pre-exec: "module load CUDA/12.3"

``slurm-job.vk.io/flags`` is passed to the scheduler, so it is where node count, GPUs per node
and walltime go. ``slurm-job.vk.io/pre-exec`` runs on the compute node before the container
starts, which is where module loads and network setup belong.

The pod's ``image`` then points at the ``.sif`` on the HPC filesystem rather than at a
registry.

Running itwinai this way
------------------------

The pattern that works today is to start a **Ray cluster whose workers are offloaded to HPC**,
then submit itwinai to it exactly as in :doc:`cloud-ray`. The KubeRay operator runs in
Kubernetes, the head pod can stay local, and the GPU worker pods land on the supercomputer. As
far as itwinai is concerned there is a Ray cluster with GPUs, so ``strategy=ddp`` becomes
:class:`~itwinai.torch.distributed.RayDDPStrategy` and nothing in the configuration changes.

A worked example, with a real KubeRay values file, is in
:doc:`../tutorials/distrib-ml/kuberay-setup-tutorial`. It runs on a Kubernetes cluster at GRNET
offloading to Vega, using the
`hython itwinai plugin <https://github.com/interTwin-eu/hython-itwinai-plugin>`_.

.. note::

   That tutorial is site-specific by necessity: the ``.sif`` path, the network interface for
   NCCL and the ``pre-exec`` commands differ at every site. Read it as a worked example rather
   than a recipe, and expect to rewrite the site-dependent parts.

Status
------

This mode is the least turnkey of the three. It is in production use within the interTwin
project and the pieces are proven, but setting it up at a new site means working with that
site. If you are evaluating it, start from :doc:`cloud-ray` on a plain Kubernetes cluster,
confirm your use case runs there, and add interLink afterwards: the itwinai side is unchanged
and only the worker pods move.
