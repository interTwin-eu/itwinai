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

Launch training using `itwinai` and the training configuration:

```bash
cd use-cases/3dgan
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline

# Or better:
micromamba run -p ../../.venv-pytorch/ torchrun --nproc_per_node gpu \
    itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline
```

To visualize the logs with MLFLow, if you set a local path as tracking URI,
run the following in the terminal:

```bash
micromamba run -p ../../.venv-pytorch mlflow ui --backend-store-uri LOCAL_TRACKING_URI
```

And select the "3DGAN" experiment.

## Inference

Disclaimer: the following is preliminary and not 100% ML/scientifically sound.

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

    ```bash
    python create_inference_sample.py
    ```

3. Run inference command. This will generate a `3dgan-generated-data`
folder containing generated particle traces in form of torch tensors
(.pth files) and 3D scatter plots (.jpg images).

    ```bash
    itwinai exec-pipeline --config config.yaml --pipe-key inference_pipeline
    ```

The inference execution will produce a folder called
`3dgan-generated-data` containing
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

However, if `aggregate_predictions` in the `ParticleImagesSaver` step is set to `True`,
only one pickled file will be generated inside `3dgan-generated-data` folder.
Notice that multiple inference calls will create new files under `3dgan-generated-data` folder.

With fields overriding:

```bash
# Override variables
export CERN_DATA_ROOT="../.."  # data root
export TMP_DATA_ROOT=$CERN_DATA_ROOT
export CERN_CODE_ROOT="." # where code and configuration are stored
export MAX_DATA_SAMPLES=20000 # max dataset size
export BATCH_SIZE=1024 # increase to fill up GPU memory
export NUM_WORKERS_DL=4 # num worker processes used by the dataloader to pre-fetch data
export AGGREGATE_PREDS="true" # write predictions in a single file
export ACCELERATOR="gpu" # choose "cpu" or "gpu"
export STRATEGY="auto" # distributed strategy
export DEVICES="0," # GPU devices list


itwinai exec-pipeline --print-config --config $CERN_CODE_ROOT/config.yaml \
    --pipe-key inference_pipeline \
    -o dataset_location=$CERN_DATA_ROOT/exp_data \
    -o logs_dir=$TMP_DATA_ROOT/ml_logs/mlflow_logs \
    -o distributed_strategy=$STRATEGY \
    -o devices=$DEVICES \
    -o hw_accelerators=$ACCELERATOR \
    -o checkpoints_path=\\$TMP_DATA_ROOT/checkpoints \
    -o inference_model_uri=$CERN_CODE_ROOT/3dgan-inference.pth \
    -o max_dataset_size=$MAX_DATA_SAMPLES \
    -o batch_size=$BATCH_SIZE \
    -o num_workers_dataloader=$NUM_WORKERS_DL \
    -o inference_results_location=$TMP_DATA_ROOT/3dgan-generated-data \
    -o aggregate_predictions=$AGGREGATE_PREDS
```

### Docker image

Build from project root with

```bash
# Local
docker buildx build -t itwinai:0.0.1-3dgan-0.1 -f use-cases/3dgan/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 -f use-cases/3dgan/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1
```

You can run inference from wherever a sample of H5 files is available
(folder called `exp_data/`'):

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
docker run -it --rm --name running-inference -v "$PWD":/tmp/data ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1
```

This command will store the results in a folder called `3dgan-generated-data`:

```text
├── $PWD
|   ├── 3dgan-generated-data
|   │   ├── energy=1.296749234199524&angle=1.272539496421814.pth
|   │   ├── energy=1.296749234199524&angle=1.272539496421814.jpg
...
|   │   ├── energy=1.664689540863037&angle=1.4906378984451294.pth
|   │   ├── energy=1.664689540863037&angle=1.4906378984451294.jpg
```

To override fields in the configuration file at runtime, you can use the `-o`
flag. Example: `-o path.to.config.element=NEW_VALUE`.

Please find a complete exampled below, showing how to override default configurations
by setting some env variables:

```bash
# Override variables
export CERN_DATA_ROOT="/usr/data" 
export CERN_CODE_ROOT="/usr/src/app"
export MAX_DATA_SAMPLES=10 # max dataset size
export BATCH_SIZE=64 # increase to fill up GPU memory
export NUM_WORKERS_DL=4 # num worker processes used by the dataloader to pre-fetch data
export AGGREGATE_PREDS="true" # write predictions in a single file
export ACCELERATOR="gpu" # choose "cpu" or "gpu"

docker run -it --rm --name running-inference \
-v "$PWD":/usr/data ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 \
/bin/bash -c "itwinai exec-pipeline \
    --print-config --config $CERN_CODE_ROOT/config.yaml \
    --pipe-key inference_pipeline \
    -o dataset_location=$CERN_DATA_ROOT/exp_data \
    -o logs_dir=$TMP_DATA_ROOT/ml_logs/mlflow_logs \
    -o distributed_strategy=$STRATEGY \
    -o devices=$DEVICES \
    -o hw_accelerators=$ACCELERATOR \
    -o checkpoints_path=\\$TMP_DATA_ROOT/checkpoints \
    -o inference_model_uri=$CERN_CODE_ROOT/3dgan-inference.pth \
    -o max_dataset_size=$MAX_DATA_SAMPLES \
    -o batch_size=$BATCH_SIZE \
    -o num_workers_dataloader=$NUM_WORKERS_DL \
    -o inference_results_location=$TMP_DATA_ROOT/3dgan-generated-data \
    -o aggregate_predictions=$AGGREGATE_PREDS "
```

#### How to fully exploit GPU resources

Keeping the example above as reference, increase the value of `BATCH_SIZE` as much as possible
(just below "out of memory" errors). Also, make sure that `ACCELERATOR="gpu"`. Also, make sure
to use a dataset large enough by changing the value of `MAX_DATA_SAMPLES` to collect meaningful
performance data. Consider that each H5 file contains roughly 5k items, thus setting
`MAX_DATA_SAMPLES=10000` should be enough to use all items in each input H5 file.

You can try:

```bash
export MAX_DATA_SAMPLES=10000 # max dataset size
export BATCH_SIZE=1024 # increase to fill up GPU memory
export ACCELERATOR="gpu
```

### Singularity

Run Docker container with Singularity:

```bash
singularity run --nv -B "$PWD":/usr/data docker://ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 /bin/bash -c \
"cd /usr/src/app && itwinai exec-pipeline --config config.yaml --pipe-key inference_pipeline"
```

Example with overrides (as above for Docker):

```bash
# Override variables
export CERN_DATA_ROOT="/usr/data" 
export CERN_CODE_ROOT="/usr/src/app"
export MAX_DATA_SAMPLES=10 # max dataset size
export BATCH_SIZE=64 # increase to fill up GPU memory
export NUM_WORKERS_DL=4 # num worker processes used by the dataloader to pre-fetch data
export AGGREGATE_PREDS="true" # write predictions in a single file
export ACCELERATOR="gpu" # choose "cpu" or "gpu"

singularity run --nv -B "$PWD":/usr/data docker://ghcr.io/intertwin-eu/itwinai:0.0.1-3dgan-0.1 /bin/bash -c \
"cd /usr/src/app && itwinai exec-pipeline \
    --print-config --config $CERN_CODE_ROOT/config.yaml \
    --pipe-key inference_pipeline \
    -o dataset_location=$CERN_DATA_ROOT/exp_data \
    -o logs_dir=$TMP_DATA_ROOT/ml_logs/mlflow_logs \
    -o distributed_strategy=$STRATEGY \
    -o devices=$DEVICES \
    -o hw_accelerators=$ACCELERATOR \
    -o checkpoints_path=\\$TMP_DATA_ROOT/checkpoints \
    -o inference_model_uri=$CERN_CODE_ROOT/3dgan-inference.pth \
    -o max_dataset_size=$MAX_DATA_SAMPLES \
    -o batch_size=$BATCH_SIZE \
    -o num_workers_dataloader=$NUM_WORKERS_DL \
    -o inference_results_location=$TMP_DATA_ROOT/3dgan-generated-data \
    -o aggregate_predictions=$AGGREGATE_PREDS "
```
