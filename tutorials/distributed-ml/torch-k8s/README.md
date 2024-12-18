# Distributed ML on Kubernetes

## Kybeflow's Training Operator

First, install kubeflow [training operator](https://www.kubeflow.org/docs/components/training/installation/).
The Python SDK are not needed for this tutorial.

Example for `v1.8.1`:

```bash
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"

kubectl get pods -n kubeflow
```
