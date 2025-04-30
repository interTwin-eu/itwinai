# Running the itwinai pipeline with KubeRay

This tutorial shows how to set up a KubeRay cluster and run the itwinai training or hyperparameter
optimization (HPO) pipeline using the [hython-itwinai-plugin](https://github.com/interTwin-eu/hython-itwinai-plugin).

- `grnet` will refer to the server from where we start and install the KubeRay cluster.
- `vega` is the supercomputer, its [resources](https://doc.vega.izum.si/general-spec/) are
  accessed from the KubeRay cluster via interlink.

The kubeRay cluster runs its pods on `vega` resources with [interLink](https://github.com/interTwin-eu/interLink).

For more background on Ray job submission and KubeRay, see:

- [KubeRay overview](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) – Setting up and managing Ray on
  Kubernetes.
- [Ray job submission guide](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) –
  Explains how to submit jobs to a Ray cluster.
- [Job submission quickstart](
    https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html
  ) – A hands-on walkthrough to get started quickly.

## Prerequisites

Ensure you have the singularity container file in an accessible location on `vega`.

On `vega`, you can pull the singularity container image of the hython plugin with:

```bash
singularity pull --force hython:sif docker://ghcr.io/intertwin-eu/hython-itwinai-plugin:<tag>
```

This will create a file called `hython:sif` in the current directory.
Check
[available hython images](https://github.com/interTwin-eu/hython-itwinai-plugin/pkgs/container/hython-itwinai-plugin)
for the right tag.

## 0. Connect to ray server (e.g. grnet) via SSH

Request access to the server first.
KubeRay creates a number of pods, which are transparently offloaded to `vega` through interLink.
The KubeRay cluster will access `vega` resources via [interlink](https://github.com/intertwin-eu/interlink).

Get into superuser shell and navigate to a work directory:

```bash
sudo su
cd .interlink # for grnet instance
```

## 1. Start the Ray cluster

To start the kubernetes cluster, you need to create a values file, e.g. `<your-name>_raycluster.yaml`.
You can copy the content of the `raycluster_example.yaml` in this directory for that.

Based on your needs, edit the values file to ensure it points to the right `sif` file:

```yaml
image:
  # TODO Change to the path where your singularity container file resides. (example is for file named hython:sif)
  repository: <path>/hython
  tag: sif
  pullPolicy: IfNotPresent
# TODO Edit resources as needed (e.g. increase number and resources per head/worker pod)
```

Then run:

```bash
helm upgrade --install raycluster kuberay/ray-cluster --version 1.2.2 --values <your-name>_raycluster.yaml
```

This will start the kuberay cluster based on the provided values file.

You can check the status of the current pods with "raycluster" in their name with:

```bash
kubectl get pods | grep raycluster
```

As the pods need to allocate jobs on `vega`, you will have to wait a few minutes before the cluster is ready for
submissions.
The pods are ready when each pod displays _1/1_ and _Running_.

> [!WARNING]
> In the end, remember to shutdown your raycluster when it is no longer in use, so it does not block `vega` resources.
> See [3. Shutting down and deleting the KubeRay cluster](3-shutting-down-and-deleting-the-kuberay-cluster)

## 2. Submit to the KubeRay cluster

To submit a ray job to the KubeRay cluster, run the following command from e.g. `vega`:

```bash
ray job submit --address <address> --working-dir <cwd> -- <command>
```

To start the hython training pipeline from the hython-itwinai-plugin directory, the full submission command would be:

```bash
ray job submit --address <address> --working-dir configuration_files/ -- itwinai exec-pipeline --config-name vega_training +pipe_key=training
```

> [!NOTE]
> The address is not public information and therefore not revealed here, ask one of the contributors in case you don't
> have it.
> It is the public address of the Ray head node. In this setup the IP is exposed via Ingress.
> But this can very for different setups.

To log to the intertwin MLflow server, overwrite the `tracking_uri` and prefix your authentication environment
variables.
E.g., for running the HPO pipeline of the hython-itwinai-plugin, run:

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
> Make sure `experiment_name` is set to a unique name as your job will fail if the name is already in use.
> In case someone else created an experiment with the same name, it will fail with `permission denied`.
> You have to create an account [here](http://mlflow.intertwin.fedcloud.eu/) first,
> then use your email as the username.

## 3. Shutting down and deleting the KubeRay cluster

When you are done:

```bash
helm delete raycluster
```

This frees the associated `vega` resources.
