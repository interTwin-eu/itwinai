# Starting a Ray cluster on HPC from Kubernetes for HPO and distributed ML training

**Author(s)**: Linus Eickhoff (CERN), Matteo Bunino (CERN)

This tutorial demonstrates how to set up a KubeRay cluster and run the itwinai training or hyperparameter optimization (HPO)
pipeline using the [hython-itwinai-plugin](https://github.com/interTwin-eu/hython-itwinai-plugin).

In this tutorial, we'll use a Kubernetes cluster hosted on `grnet` (a cloud provider) to run KubeRay, which will then access
computing resources on the `Vega` supercomputer (an HPC system) through interLink. While we use specific infrastructure in
this example, the concepts apply to any Kubernetes cluster accessing HPC resources.

In this guide we use the following as `cloud` and `HPC` resources:

- `grnet` is the `cloud` server where we start and install the KubeRay cluster ([grnet homepage](https://grnet.gr/en/))
- `Vega` is the `HPC` environment whose [resources](https://doc.vega.izum.si/general-spec/) are accessed from the KubeRay
    cluster via interlink ([Vega homepage](https://izum.si/en/vega-en/))

The KubeRay cluster runs its pods on `HPC` resources using [interLink](https://github.com/interTwin-eu/interLink).

For additional background on Ray job submission and KubeRay, refer to:

- [KubeRay overview](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) –
    Setting up and managing Ray on Kubernetes
- [Ray job submission guide](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) –
    Explains how to submit jobs to a Ray cluster
- [Job submission quickstart](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html)
    – A hands-on walkthrough to get started quickly

## Prerequisites

Make sure you have the singularity container file in an accessible location on `HPC`.

To pull the singularity container image of the hython plugin on `HPC`, run:

```bash
singularity pull --force hython:sif docker://ghcr.io/intertwin-eu/hython-itwinai-plugin:<tag>
```

This creates a file named `hython:sif` in the current directory.
Check the [available hython images](https://github.com/interTwin-eu/hython-itwinai-plugin/pkgs/container/hython-itwinai-plugin)
for the appropriate tag.

## Connect to cloud server via SSH

First, request access to the cloud server.
KubeRay creates multiple pods that are transparently offloaded to `HPC` through interLink.
The KubeRay cluster accesses `HPC` resources via [interlink](https://github.com/intertwin-eu/interlink).

In the specific case of using `grnet`, switch to superuser shell and navigate to the work directory:

```bash
# for grnet instance
sudo su
cd .interlink
```

## Start the Ray cluster

To start the Kubernetes cluster, create a values file (e.g., `<your-name>_raycluster.yaml`).
You can use the `raycluster_example.yaml` in this directory as a template.

Edit the values file (`raycluster_example.yaml`) to ensure it points to the correct `sif` file:

```yaml
image:
    # TODO Change to the path where your singularity container file resides. (example is for file named hython:sif)
    repository: <path>/hython
    tag: sif
    pullPolicy: IfNotPresent
# TODO Edit resources as needed (e.g. increase number and resources per head/worker pod)
```

To get an overview over the available attributes for ray values files, please consult the ray documentation for the
[RayCluster Configuration](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html).

Then execute:

```bash
helm upgrade --install raycluster kuberay/ray-cluster --version 1.2.2 --values <your-name>_raycluster.yaml
```

This command starts the KubeRay cluster based on your values file.

To check the status of pods with "raycluster" in their name:

```bash
kubectl get pods | grep raycluster
```

Since the pods need to allocate jobs on `HPC`, wait a few minutes for the cluster to be ready for submissions.
The pods are ready when each pod shows _1/1_ and _Running_.

> [!WARNING]
> Remember to shut down your raycluster when it's no longer in use to free up `HPC` resources.
> See [3. Shutting down and deleting the KubeRay cluster](#shutting-down-and-deleting-the-kuberay-cluster)

## Submit to the KubeRay cluster

To submit a Ray job to the KubeRay cluster from `HPC`, run:

```bash
ray job submit --address <address> --working-dir <cwd> -- <command>
```

For example, to start the hython training pipeline from the hython-itwinai-plugin directory:

```bash
ray job submit --address <address> --working-dir configuration_files/ -- itwinai exec-pipeline --config-name vega_training +pipe_key=training
```

> [!NOTE]
> The address is not public information. Please contact one of the contributors if you don't have it.
> It represents the public address of the Ray head node, exposed via Ingress in this setup.
> Note that this configuration may vary for different setups.

To log to the intertwin MLflow server, override the `tracking_uri` and prefix your authentication environment variables.
For example, to run the HPO pipeline of the hython-itwinai-plugin:

```bash
ray job submit \
  --address <address> \
  --working-dir configuration_files/ \
  -- \
  MLFLOW_TRACKING_USERNAME=<username> \
  MLFLOW_TRACKING_PASSWORD=<password> \
  itwinai exec-pipeline \
    --config-name <config-name> \
    tracking_uri=http://mlflow.intertwin.fedcloud.eu/ \
    +pipe_key=hpo
```

> [!NOTE]
> Ensure `experiment_name` is set to a unique name, as your job will fail if the name is already in use.
> If someone else created an experiment with the same name, it will fail with `permission denied`.
> First, create an account [here](http://mlflow.intertwin.fedcloud.eu/), then use your email as the username.

## Shutting down and deleting the KubeRay cluster

When finished, run the following on the `cloud` instance:

```bash
helm delete raycluster
```

This command releases the associated `HPC` resources.
