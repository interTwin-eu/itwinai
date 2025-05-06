Distributed Machine Learning on HPC from k8s using KubeRay operator and interLink
=================================================================================

.. include:: ../../../tutorials/distributed-ml/kuberay-setup-tutorial/README.md
   :parser: myst_parser.sphinx_


raycluster_example.yaml
+++++++++++++++++++++++

This file defines the RayCluster, the file is referenced in the tutorial as the values file
used by the KubeRay operator to deploy Ray
clusters on Kubernetes.
It specifies the configuration for head and worker nodes, including resource requests,
environment variables, and startup commands.
For a full reference of supported fields and structure, see the
`Ray on Kubernetes config documentation <https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html>`_


.. literalinclude:: ../../../tutorials/distributed-ml/kuberay-setup-tutorial/raycluster_example.yaml
   :language: yaml
