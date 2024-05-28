# Noise Simulation for Gravitational Waves Detector (Virgo)

## Installation

Before continuing, install the required libraries in the pre-existing itwinai environment.

```bash
pip install -r requirements.txt
```

## Training

You can run the whole pipeline in one shot, including dataset generation, or you can
execute it from the second step (after the synthetic dataset have been generated).

```bash
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline

# Run from the second step (use python-like slicing syntax).
# In this case, the dataset is loaded from "data/Image_dataset_synthetic_64x64.pkl"
itwinai exec-pipeline --config config.yaml --pipe-key training_pipeline --steps 1:
```
