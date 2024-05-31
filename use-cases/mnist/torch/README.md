# Pure torch example on MNIST dataset

## Training

```bash
# Download dataset and exit
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps dataloading_step

# Run the whole training pipeline
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline 
```

View training logs on MLFLow server (if activated from the configuration):

```bash
mlflow ui --backend-store-uri mllogs/mlflow/
```

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
    itwinai exec-pipeline --config config.yaml --pipe-key inference_pipeline 
    ```

Note the same entry point as for training.

## Docker image

Build from project root with

```bash
# Local
docker buildx build -t itwinai:0.0.1-mnist-torch-0.1 -f use-cases/mnist/torch/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:0.0.1-mnist-torch-0.1 -f use-cases/mnist/torch/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai:0.0.1-mnist-torch-0.1
```

### Training with Docker container

```bash
docker run -it --rm --name running-inference \
    -v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai:0.01-mnist-torch-0.1 \
    /bin/bash -c "itwinai exec-pipeline --print-config \
    --config /usr/src/app/config.yaml \
    --pipe-key training_pipeline \
    -o dataset_root=/usr/data/mnist-dataset "
```

### Inference with Docker container

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
docker run -it --rm --name running-inference \
    -v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai:0.01-mnist-torch-0.1 \
    /bin/bash -c "itwinai exec-pipeline --print-config \
    --config /usr/src/app/config.yaml \
    --pipe-key inference_pipeline \
    -o test_data_path=/usr/data/mnist-sample-data \
    -o inference_model_mlflow_uri=/usr/src/app/mnist-pre-trained.pth \
    -o predictions_dir=/usr/data/mnist-predictions "
```

This command will store the results in a folder called "mnist-predictions":

```text
├── $PWD
│   ├── mnist-predictions
|   │   ├── predictions.csv
```
