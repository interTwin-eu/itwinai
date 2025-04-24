# Containerization for Vega (Kubernetes + Singularity)

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
Then SSH on the server, from which to launch the KubeRay cluster, which then manages
jobs via `interlink`.

Get into root shell and navigate to a work directory:

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

You can check the status of the current pods with:

```bash
kubectl get pods | grep raycluster
```

As the pods need to allocate jobs on vega, you will have to wait a few minutes before the cluster is ready for submits.
The pods are ready when each pod displays _1/1_ and _Running_.

> [!CAUTION]
> Remember to shutdown your raycluster, so it does not block vega resources when not used anymore:

```bash
helm delete raycluster
```

## 2. Submit to the Cluster

To submit a ray job to the cluster, run the following command from e.g. vega:

```bash
ray job submit --address <address> --workdir <path> -- <command>
```

To start the hython training pipeline, the submit would be:

```bash
ray job submit --address <address> --workdir . -- itwinai exec-pipeline --config-path configuration_files --config-name training
```

> [!NOTE]
> The address is not public information and therefor not revealed here, ask one of the contributors in case you miss it.
