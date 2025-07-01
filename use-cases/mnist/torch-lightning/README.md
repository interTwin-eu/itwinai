# Torch Lightning example on MNIST dataset

**Integration author(s)**: Matteo Bunino (CERN)

## Training

```bash
# Download dataset and exit: only run first step in the pipeline (index=0)
itwinai exec-pipeline +pipe_key=training_pipeline +pipe_steps=[0]

# Run the whole training pipeline
itwinai exec-pipeline +pipe_key=training_pipeline 
```

View training logs on MLFLow server (if activated from the configuration):

```bash
mlflow ui --backend-store-uri mllogs/mlflow/
```
