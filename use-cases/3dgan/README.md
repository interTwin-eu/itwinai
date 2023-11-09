# 3DGAN use case

First of all, from the repository root, create a torch environment:

```bash
make torch-gpu
```

Now, install custom requirements for 3DGAN:

```bash
micromamba activate ./.venv-pytorch
cd use-cases/3dgan
pip install -r requirements.txt
```

**NOTE**: Python commands below assumed to be executed from within the
micromamba virtual environment.

## Training

At CERN, use the dedicated configuration file:

```bash
cd use-cases/3dgan
python train.py -p cern-pipeline.yaml

# Or better:
micromamba run -p ../../.venv-pytorch/ torchrun train.py -p cern-pipeline.yaml
```

Anywhere else, use the general purpose training configuration:

```bash
cd use-cases/3dgan
python train.py -p pipeline.yaml

# Or better:
micromamba run -p ../../.venv-pytorch/ torchrun train.py -p pipeline.yaml
```

To visualize the logs with MLFLow run the following in the terminal:

```bash
micromamba run -p ../../.venv-pytorch mlflow ui --backend-store-uri ml_logs/mlflow_logs
```

And select the "3DGAN" experiment.

## Inference

The following is preliminary and not 100% ML/scientifically sound.

1. As inference dataset we can reuse training/validation dataset,
for instance the one downloaded from Google Drive folder: if the
dataset root folder is not present, the dataset will be downloaded.
The inference dataset is a set of H5 files stored inside `exp_data`
sub-folders:

    ```text
    ├── exp_data
    │   ├── data
    |   │   ├── file_0.h5
    |   │   ├── file_1.h5
    ...
    |   │   ├── file_N.h5
    ```

2. As model, if a pre-trained checkpoint is not available,
we can create a dummy version of it with:

    ```python
    import torch
    from model import ThreeDGAN
    # Same params as in the training config file!
    my_gan = ThreeDGAN()
    torch.save(my_gan, '3dgan-inference.pth')
    ```

3. Run inference command. This will generate a "3dgan-generated"
folder containing generated particle traces in form of torch tensors
(.pth files) and 3D scatter plots (.jpg images).

    ```bash
    python train.py -p inference-pipeline.yaml
    ```

Note the same entry point as for training.

The inference execution will produce a folder called
"3dgan-generated-data" containing
generated 3D particle trajectories (overwritten if already
there). Each generated 3D image is stored both as a
torch tensor (.pth) and 3D scatter plot (.jpg):

```text
├── 3dgan-generated-data
|   ├── energy=1.296749234199524&angle=1.272539496421814.pth
|   ├── energy=1.296749234199524&angle=1.272539496421814.jpg
...
|   ├── energy=1.664689540863037&angle=1.4906378984451294.pth
|   ├── energy=1.664689540863037&angle=1.4906378984451294.jpg
```

### Docker image

Build from project root with

```bash
# Local
docker buildx build -t itwinai-mnist-torch-inference -f use-cases/3dgan/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai-3dgan-inference:0.0.1 -f use-cases/3dgan/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai-3dgan-inference:0.0.1
```

From wherever a sample of MNIST jpg images is available
(folder called 'mnist-sample-data/'):

```text
├── $PWD    
|   ├── exp_data
|   │   ├── data
|   |   │   ├── file_0.h5
|   |   │   ├── file_1.h5
...
|   |   │   ├── file_N.h5
```

```bash
docker run -it --rm --name running-inference -v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai-3dgan-inference:0.0.1
```

This command will store the results in a folder called "3dgan-generated-data":

```text
├── $PWD
|   ├── 3dgan-generated-data
|   │   ├── energy=1.296749234199524&angle=1.272539496421814.pth
|   │   ├── energy=1.296749234199524&angle=1.272539496421814.jpg
...
|   │   ├── energy=1.664689540863037&angle=1.4906378984451294.pth
|   │   ├── energy=1.664689540863037&angle=1.4906378984451294.jpg
```
