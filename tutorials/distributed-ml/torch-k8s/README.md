# Distributed ML on Kubernetes

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

1. The PyTorchJob sets up the correct envirnment variables that will be picked up by `torchrun`
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
