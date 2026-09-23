# Tutorial: pushing and pulling models with the RI-SCALE Model Hub

**Author(s)**: Rakesh Sarma (FZJ)

This tutorial trains a small model with `itwinai`'s `TorchTrainer`, pushes its checkpoint to
the Model Hub, then pulls that same checkpoint back and runs inference on it with
`TorchPredictor`. Both steps run as itwinai pipelines via the `itwinai exec-pipeline` CLI.

For the full explanation of the push/pull feature with the AI Model Hub, please look at the
detailed documentation in [Accessing models from the RI-SCALE Model Hub](../../../docs/how-it-works/model-hub/explain_model_hub.rst).

## Setup

1. Install `torch` and `modelhub` extras:

```bash
uv pip install -e ".[torch,dev,modelhub]" --no-cache-dir
```

2. `cd` into this tutorial's directory (`tutorials/model-hub/torch-tutorial-model-hub`) --
`itwinai exec-pipeline` looks for `config.yaml` and `.env` in the current directory.

3. Create a `.env` file in this directory with `HYPHA_SERVER_URL` and `HYPHA_TOKEN` (see
[this page](https://github.com/RI-SCALE/ai-model-hub-example/blob/main/.env.example) for how to
obtain them). Both push and pull need internet access; push additionally needs to reach GitHub
to download its upload script.

`synthetic_data.py` defines `SanityCheckModel` and two synthetic datasets
(`SyntheticCheckpointDataset` for training, `SyntheticInferenceDataset` for inference).
`data.py` wraps those datasets in two pipeline steps used by `config.yaml`.

## Part 1: train and push

```bash
itwinai exec-pipeline +pipe_key=training_pipeline
```

## Part 2: pull and run inference

```bash
itwinai exec-pipeline +pipe_key=inference_pipeline
```

Run this only after Part 1 has completed successfully -- it pulls the same `model_id` that
Part 1 pushed.

## Requirements

Internet access is required for both steps.
