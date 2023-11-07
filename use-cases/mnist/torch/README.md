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

### Docker image

Build from project root with

```bash
# Local
docker buildx build -t itwinai-mnist-torch-inference -f use-cases/mnist/torch/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai-mnist-torch-inference:0.0.1 -f use-cases/mnist/torch/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai-mnist-torch-inference:0.0.1
```

From wherever a sample of MNIST jpg images is available
(folder called 'mnist-sample-data/'):

```text
├── $PWD
│   ├── mnist-sample-data
|   │   ├── digit_0.jpg
|   │   ├── digit_1.jpg
|   │   ├── digit_2.jpg
...
|   │   ├── digit_N.jpg
```

```bash
docker run -it --rm --name running-inference -v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai-mnist-torch-inference:0.0.1
```

This command will store the results in a folder called "mnist-predictions":

```text
├── $PWD
│   ├── mnist-predictions
|   │   ├── predictions.csv
```
