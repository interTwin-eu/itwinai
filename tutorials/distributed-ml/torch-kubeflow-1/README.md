# Distributed Machine Learning on Kubernetes with Kubeflow

**Author(s)**: Matteo Bunino (CERN)

This tutorial shows how to run distributed machine learning on Kubernetes when an HPC cluster managed
by SLURM is unavailable. It provides a flexible solution for both on-premises and cloud-based Kubernetes
deployments, including access to GPU nodes for scalable, high-performance training workloads.
It demonstrates running distributed machine learning (ML) on Kubernetes using
Kubeflow's [training operator](https://www.kubeflow.org/docs/components/training/overview/)
for PyTorch and itwinai's `TorchTrainer`.

We will only use `kubectl` and pod manifests to launch jobs, requiring minimal setup beyond
access to a Kubernetes cluster with a few nodes. The Python SDK is beyond this guide's scope,
but you can explore Kubeflow's
[getting started tutorial](https://www.kubeflow.org/docs/components/training/getting-started/#getting-started-with-pytorchjob)
for more details.

## Installing Kubeflow's Training Operator

First, install the [training operator](https://www.kubeflow.org/docs/components/training/installation/).
Python SDK is not needed for this tutorial.

Example for `v1.8.1`:

```bash
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
```

Check that the training operator is running:

```bash
kubectl get pods -n kubeflow
```

Output should include:

```text
NAME                                 READY   STATUS    RESTARTS   AGE
training-operator-6f4d5d95f8-spfgx   1/1     Running   0          11d
```

Before proceeding, familiarize yourself with how
[PyTorchJob](https://www.kubeflow.org/docs/components/training/reference/distributed-training/#distributed-training-for-pytorch)
works. In brief:

1. The PyTorchJob sets environment variables for `torchrun`.
1. The Python script should be invoked using `torchrun` in the pod manifest.
The `torchrun` CLI will make sure that the correct number of worker processes (i.e., replicas of your Python process)
per pod are spawned. Example:

    ```yaml
    containers:
    - name: pytorch
        image: registry.cern.ch/itwinai/dist-ml/itwinai-slim:0.0.7
        command:
        - "torchrun"
        - "/app/main.py"
    ```

Set the number of processes per node using
[`nProcPerNode`](https://github.com/kubeflow/training-operator/blob/69094e16309382d929606f8c5ce9a9d8c00308b1/pkg/apis/kubeflow.org/v1/pytorch_types.go#L95)
. It maps to `torchrun`'s `--nproc-per-node`.

### Creating a PyTorchJob

To submit a job, use:

```bash
kubectl create -n kubeflow -f job-manifest.yaml
```

When creating a PyTorchJob, the Worker pods will wait for the Master to be created
first. To manage both Master and Worker pods use:

```bash
# Inspect some pods
kubectl describe pod torchrun-cpu-worker-0 -n kubeflow
kubectl describe pod torchrun-cpu-master-0 -n kubeflow

# Get the logs from the pods
kubectl logs torchrun-cpu-master-0 -n kubeflow
kubectl logs torchrun-cpu-worker-0 -n kubeflow
```

Delete all PyTorchJobs:

```bash
kubectl delete --all pytorchjobs -n kubeflow
```

To remove the training operator:

```bash
kubectl delete deployment training-operator -n kubeflow
```

## Distributed Training on CPU

To get started with distributed ML using Kubeflow and itwinai, a GPU cluster is not required.
The PyTorchJob manifest for CPU-based training is defined in `cpu.yaml`. First, build and
push a Docker container using the provided `Dockerfile`, then update the manifest with
your container's image name.

The manifest sets `nProcPerNode: "2"`, which specifies two worker processes per pod.
Adjust this for different levels of parallelism, corresponding to the
[`--nproc-per-node`](https://pytorch.org/docs/stable/elastic/run.html#usage) flag of `torchrun`.

There are two levels of parallelism:

- **Pod-level parallelism**: Controlled by the number of `replicas` in the PyTorchJob.
- **Process-level parallelism**: Controlled by `nProcPerNode` for multiple subprocesses per pod.

Using `nProcPerNode > 1` allows two levels of parallelism. Each pod runs on a different node,
spawning as many processes as hardware accelerators (like GPUs). Parallelism is:
`nProcPerNode * TOTAL_PODS`.

Alternatively, setting `nProcPerNode: "1"` uses pod replicas to control parallelism,
with one pod per distributed ML worker. However, this may be less efficient (e.g., when
using persistent storage).

## Distributed Training on GPU

> [!NOTE]
> This part has not been extensively tested and is still under development.

To access GPU nodes, add the following request to the containers spec
in the job manifest:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

Example:

```yaml
  ...
  containers:
  - name: pytorch
      image: registry.cern.ch/itwinai/dist-ml/itwinai-slim:0.0.7
      command:
      - "torchrun"
      - "/app/main.py"
      resources:
        limits:
          nvidia.com/gpu: 1
```

To allocate a full node to a pod, set the number of requested GPUs
equal to the number of GPUs available on the node, and adjust the number of replicas,
`nProcPerNode`, accordingly.
