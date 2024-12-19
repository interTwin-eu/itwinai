# Distributed ML on Kubernetes with Kubeflow

**Author(s)**: Matteo Bunino (CERN)

In this tutorial we see how to run distributed machine learning (ML) on Kubernetes using
Kubeflow's [training operator](https://www.kubeflow.org/docs/components/training/overview/)
for PyTorch and itwinai's `TorchTrainer`.

In this guide we will show you how to launch jobs only by using `kubectl` and pod manifests.
Therefore, you won't need much more than being able to access a kubernetes cluster with few
nodes. We won't cover Python SDK, which can be found in
[this nice tutorial](https://www.kubeflow.org/docs/components/training/getting-started/#getting-started-with-pytorchjob)
from Kubeflow.

## Kybeflow's Training Operator

First, install kubeflow [training operator](https://www.kubeflow.org/docs/components/training/installation/).
The Python SDK are not needed for this tutorial.

Example for `v1.8.1`:

```bash
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```

Check that the training operator is running:

```bash
$ kubectl get pods -n kubeflow
NAME                                 READY   STATUS    RESTARTS   AGE
...
training-operator-6f4d5d95f8-spfgx   1/1     Running   0          11d
```

Before continuing further, take some time to familiarize with how a
[PyTorchJob](https://www.kubeflow.org/docs/components/training/reference/distributed-training/#distributed-training-for-pytorch)
works. Instuitively:

1. The PyTorchJob sets up the correct envirnment variables that will be picked up by `torchrun`.
1. Your Python script should be executed prepending `torchrun` to it in the pod manifest.
The `torchrun` CLI will make sure that the correct number of worker processes (i.e., replicas of your Python process)
per pod is spawned. Example:

    ```yaml
    ...
    containers:
    - name: pytorch
        image: registry.cern.ch/itwinai/dist-ml/itwinai-slim:0.0.7
        command:
        - "torchrun"
        - "/app/main.py"
    ```

The number of workers per pod is set by the
[`nProcPerNode`](https://github.com/kubeflow/training-operator/blob/69094e16309382d929606f8c5ce9a9d8c00308b1/pkg/apis/kubeflow.org/v1/pytorch_types.go#L95)
field in the PyTorchJob manifest, which is mapped to `torchrun`'s `--nproc-per-node`.

Now, create a PyTorchJob:

```bash
kubectl create -n kubeflow -f job-manifest.yaml
```

When creating a PyTorchJob, the Worker pods will wait for the Master to be online
first. To manage both Master and Worker pods you can use:

```bash
# Inspect some pods
kubectl describe pod torchrun-cpu-worker-0 -n kubeflow
kubectl describe pod torchrun-cpu-master-0 -n kubeflow

# Get the logs from the commands run the pods
kubectl logs torchrun-cpu-master-0 -n kubeflow
kubectl logs torchrun-cpu-worker-0 -n kubeflow

# To delete all existing pytorchjobs
kubectl delete --all pytorchjobs -n kubeflow
```

To delete the deployment of the training operator:

```bash
kubectl delete deployment training-operator -n kubeflow
```

## Distributed training on CPU

To get familiar with distrubuted ML with Kubeflow and itwinai you don't need access to
a Kubernets cluster with GPU nodes.

The PyTorchJob manifest for this can be found in `cpu.yaml`.
Remeber to first create a Docker container using the provided `Dockerfile`, push it to your
preferred containers registry, and update the manifest file with the correct name of the image.