# Offloading through interLink

This folder contains kubernetes pod examples to be used alongside with
[interLink](https://github.com/interTwin-eu/interLink),
to offload computation to remote HPC providers.

To use these pods, you will need to install `kubectl` and setup a `kubeconfig`.

## Manage pod

```bash
# A pod needs to be deleted before re-submitting another one with the same name
kubectl delete pod POD_NAME
# Alternatively
kubectl apply --overwrite --force -f test.yaml

# Submit pod
kubectl apply -f my-pod.yaml

# Get status
kubectl get nodes
kubectl get pods

# Get pod STDOUT
kubectl logs --insecure-skip-tls-verify-backend POD_NAME
```

## Pod annotations

Allocate resources through SLURM:

```yaml
slurm-job.vk.io/flags: "-p gpu --gres=gpu:1 --ntasks-per-node=1 --nodes=1"
```

On some HPC system it may be needed to download the docker
container before submitting the offloaded job. T0 do so, you can use the
following annotation:

```yaml
slurm-job.vk.io/pre-exec: "singularity pull /ceph/hpc/data/st2301-itwin-users/itwinaiv6_1.sif docker://ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.2"
```

IMPORTANT: add this annotation only once, when the image is not there.

## Request resources: requests and limits

GPUs shall be requested in the SLURM annotation `slurm-job.vk.io/flags` as described above.
Number of CPUs and memory, on the other hand, is defined using pod limits and requests.

For instance:

```yaml
resources:
    limits:
        cpu: "48"
        memory: 150Gi
    requests:
        cpu: "4"
        memory: 20Gi
```

`limits` define upped bounds, whereas `requests` lower bounds.
See [here](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-resource-requests-and-limits)
to know more.

## Node selector

To select to which remote system to offload, change the value in the node selector:

```yaml
nodeSelector:
    kubernetes.io/hostname: vega-new-vk
```

Additional info in [interLink](https://github.com/interTwin-eu/interLink) docs.

## Secrets

See [this guide](https://kubernetes.io/docs/tasks/inject-data-application/distribute-credentials-secure/#define-container-environment-variables-using-secret-data)
on how to set Kubernetes secretes as env variables of a container.

Example:

```bash
# Create secret to store MLFlow server credentials 
kubectl create secret generic mlflow-server --from-literal=username='XYZ' --from-literal=password='ABC'

# Inspect secrets
kubectl get secret
kubectl describe secret SECRET_NAME

# Delete secrets
kubectl delete secret SECRET_NAME
```
