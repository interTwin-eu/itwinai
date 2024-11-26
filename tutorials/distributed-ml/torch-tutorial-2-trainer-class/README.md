# Tutorial on itwinai TorchTrainer for MNIST use case

**Author(s)**: Matteo Bunino (CERN)

The code is adapted from [this example](https://github.com/pytorch/examples/blob/main/mnist/main.py).

## Run the script

```bash
python train.py

# With distributed training (interactive)
torchrun --standalone --nnodes=1 --nproc-per-node=gpu train.py --strategy ddp
```

## Analyze the logs

Analyze the logs with MLFlow:

```bash
itwinai mlflow-ui --path mllogs/mlflow
```
