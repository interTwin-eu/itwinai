# Containerization for Vega (KubeRay + Singularity)

In this tutorial, we will set up a KubeRay cluster and run HPO using
the [hython-itwinai-plugin](https://github.com/interTwin-eu/hython-itwinai-plugin).

## Prerequisites

Ensure you have the singularity container file in an accessible location on `vega`.

On Vega, you can pull the singularity container image of the hython plugin with:

```bash
singularity pull --force hython:sif docker://ghcr.io/intertwin-eu/hython-itwinai-plugin:<tag>
```

This will create a file called `hython:sif` in the current directory.
Check
[available hython images](https://github.com/interTwin-eu/hython-itwinai-plugin/pkgs/container/hython-itwinai-plugin)
for the right tag.

## 0. Connect to ray server (e.g. grnet) via SSH

Request access to the server first.
SSH into the server, which you will use to launch the KubeRay cluster.
The KubeRay cluster will use Vega resources via [interlink](https://github.com/intertwin-eu/interlink).
Then SSH on the server, from which to launch the KubeRay cluster, which then manages
jobs via `interlink`.

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

As the pods need to allocate jobs on vega, you will have to wait a few minutes before the cluster is ready for submissions.
The pods are ready when each pod displays _1/1_ and _Running_.

> [!CAUTION]
> Remember to shutdown your raycluster, so it does not block vega resources when not used anymore:

```bash
helm delete raycluster
```

## 2. Submit to the KubeRay cluster

To submit a ray job to the KubeRay cluster, run the following command from e.g. vega:

```bash
ray job submit --address <address> --working-dir <cwd> -- <command>
```

To start the hython training pipeline from the hython-itwinai-plugin directory, the full submission command would be:

```bash
ray job submit --address <address> --working-dir configuration_files/ -- itwinai exec-pipeline --config-name <config-name>
```

> [!NOTE]
> The address is not public information and therefor not revealed here, ask one of the contributors in case you miss it.

To log to the intertwin MLflow server, overwrite the `tracking_uri` and prefix your authentication environment variables:

```bash
ray job submit \
  --address <address> \
  --working-dir configuration_files/ \
  -- \
  MLFLOW_TRACKING_USERNAME=<username> \
  MLFLOW_TRACKING_PASSWORD=<password> \
  itwinai exec-pipeline \
    --config-name <config-name> \
    tracking_uri=http://mlflow.intertwin.fedcloud.eu/
```

> [!NOTE]
> Ensure to set `experiment_name` to an individual name.
> In case someone else created an experiment with the same name, it will fail with `permission denied`.
> You have to create an account [here](http://mlflow.intertwin.fedcloud.eu/) first,
> then use your email as the username.
