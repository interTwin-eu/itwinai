# Torch Lightning example on MNIST dataset

## Training

```bash
# Download dataset and exit: only run first step in the pipeline (index=0)
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 0

# Run the whole training pipeline
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline 
```

View training logs on MLFLow server (if activated from the configuration):

```bash
mlflow ui --backend-store-uri mllogs/mlflow/
```
