Cloud to HPC
============

Sometimes the orchestration belongs in Kubernetes but the compute belongs on HPC: the GPUs are
there, or the dataset is too large to move, or the allocation is already granted.
`interLink <https://github.com/interlink-hq/interLink>`_ makes that possible without running
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
registry. See :doc:`../getting-started/containers` for building it.

Running itwinai with interLink
------------------------------

Notebooks
~~~~~~~~~

In interTwin we offloaded interactive JupyterLab sessions this way: the hub stays in
Kubernetes and the single-user server runs on an HPC node. The ``jlab-slim`` image ships
``/opt/ssh/jupyterhub-singleuser``, which wraps the server in an asyncssh tunnel back to the
hub so it stays reachable from outside the HPC network. The notebook itself is the one in
:doc:`cloud-notebook`.

Ray
~~~

The pattern that works today is to start a **Ray cluster whose workers are offloaded to HPC**,
then submit itwinai to it exactly as in :doc:`cloud-ray`. The KubeRay operator runs in
Kubernetes, the head pod can stay local, and the GPU worker pods land on the supercomputer. As
far as itwinai is concerned there is a Ray cluster with GPUs, so ``strategy=ddp`` becomes
:class:`~itwinai.torch.distributed.RayDDPStrategy` and nothing in the configuration changes.
