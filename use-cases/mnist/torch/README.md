# Pure torch example on MNIST dataset

## Training

```bash
python train.py -p pipeline.yaml [-d]
```

Use `-d` flag to run only the fist step in the pipeline.

## Inference

1. Create sample dataset

    ```python
    from dataloader import InferenceMNIST
    InferenceMNIST.generate_jpg_sample('mnist-sample-data/', 10)
    ```

2. Generate a dummy pre-trained neural network

    ```python
    import torch
    from model import Net
    dummy_nn = Net()
    torch.save(dummy_nn, 'mnist-pre-trained.pth')
    ```

3. Run inference command. This will generate a "mnist-predictions"
folder containing a CSV file with the predictions as rows.

    ```bash
    python train.py -p inference-pipeline.yaml
    ```

Note the same entry point as for training.
