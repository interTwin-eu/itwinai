# Pure torch example on MNIST dataset

**Integration author(s)**: Matteo Bunino (CERN)

In this simple use case integration we demostrate how to use itwinai for a set of simple
use cases based on the popular MNIST dataset.

## Training a CNN classifier

It is possible to launch the training of a CNN classifier on the MNIST dataset using the
YAML configuration file describing the whole training workflow.

```bash
# Run the whole training pipeline
itwinai exec-pipeline --config-name config.yaml
```

Notice that the training "pipeline" starts by downloading the dataset if not available locally.
Since on some HPC systems there is no internet connection on the compute nodes, it is
advisable to run the dataloading step on the login node to download the dataset and, later,
the whole pipeline on the compute nodes. To do that, you can use the `pipe_steps` option as
below:

```bash
# Download dataset and exit
itwinai exec-pipeline --config-name config.yaml +pipe_steps=[dataloading_step]

# Run the whole pipeline
itwinai exec-pipeline --config-name config.yaml
```

> [!NOTE]
> Setting `HYDRA_FULL_ERROR=1` environment variable can be convenient when debugging errors
> that originate during the instantiation of the pipeline.

View training logs on MLFLow server (if activated from the configuration):

```bash
mlflow ui --backend-store-uri mllogs/mlflow/
```

### Hyper-parameter optimization

The CNN classifier can undergo hyper-parameter optimization (HPO) to find the hyper-parameters,
such as learning rate and batch size, that result in the best validation performances.

To do so, it is enough to correctly set the `search_space` and the `tune_config` in the trainer
configuration in the `config.yaml` file.
Please refer to the Ray's official documentation to know more about
[RunConfig](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html),
[TuneConfig](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html),
[ScalingConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html),
and [search spaces](https://docs.ray.io/en/latest/tune/api/search_space.html).

### Inference

Now you can use the trained model to make predictions on the MNIST dataset.
Notice that the inference is defined by using a different pipeline in the `config.yaml` file.
By default, the `training_pipeline` is executed, but you can run other piplines by explicitly
setting the `+pipe_key` option.

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
    itwinai exec-pipeline --config-name config.yaml +pipe_key=inference_pipeline 
    ```

Note the same entry point as for training.

## Training a GAN

In this use case you can also find an example on how to train a Generative Adversarial Network
(GAN). All you need to do is specify that you wish to use the GAN by setting the `+pipe_key`
option.

```bash
# Train a GAN
itwinai exec-pipeline --config-name config.yaml +pipe_key=training_pipeline_gan
```

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
    /bin/bash -c "itwinai exec-pipeline \
    --config-path /usr/src/app \
    +pipe_key=training_pipeline \
    dataset_root=/usr/data/mnist-dataset "
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
    /bin/bash -c "itwinai exec-pipeline \
    --config-path /usr/src/app \
    +pipe_key=inference_pipeline \
    test_data_path=/usr/data/mnist-sample-data \
    inference_model_mlflow_uri=/usr/src/app/mnist-pre-trained.pth \
    predictions_dir=/usr/data/mnist-predictions "
```

This command will store the results in a folder called "mnist-predictions":

```text
├── $PWD
│   ├── mnist-predictions
|   │   ├── predictions.csv
```
