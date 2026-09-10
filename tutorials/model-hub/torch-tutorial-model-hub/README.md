# Tutorial: pushing and pulling models with the RI-SCALE Model Hub

**Author(s)**: Rakesh Sarma (FZJ)

This tutorial shows the full implementation: train a small model with `itwinai`'s
`TorchTrainer`, push its checkpoint to the Model Hub, then pull that same checkpoint back
and run inference on it with `TorchPredictor`. Both steps run as itwinai pipelines via the
`itwinai exec-pipeline` CLI. Please note than when installing the `itwinai` environment, you
need to select the extra option `modelhub` to install the dependencies related to the AI Model
Hub.

`synthetic_data.py` defines `SanityCheckModel` (a small 3-layer CNN) and two synthetic
datasets (`SyntheticCheckpointDataset` for training, `SyntheticInferenceDataset` for
inference). The data itself is random noise -- only the tensor shapes matter here, since
the goal of this tutorial is to demonstrate the push/pull feature, not model quality.
`data.py` wraps those datasets in two small pipeline steps:
`SyntheticCheckpointDatasetSplitter` (train/val split) and `SyntheticInferenceDatasetGenerator`
(inference set).

## Part 1: train and push

```bash
itwinai exec-pipeline +pipe-key=training_pipeline
```

`config.yaml` contains all the configuration options for the trainer, predictor and the AI
Model Hub parameters that are needed to be specified.
`training_pipeline` has two steps: `SyntheticCheckpointDatasetSplitter` produces
`(train_dataset, validation_dataset)`, which the pipeline unpacks straight into
`TorchTrainer.execute(train_dataset, validation_dataset)`. The trainer step itself is built
with nested `_target_`s: `model` builds `SanityCheckModel`.

The `model_hub` block defines the parameters needed to push the model to the Model Hub. Here
`TorchTrainer.__init__` reads `getattr(self.config, "model_hub", {})` to
build its `ModelHubFeature`.

Because `model_hub.enabled: true`, every time `TorchTrainer.save_checkpoint` saves the
"best_model" checkpoint, it hands the checkpoint directory to
`ModelHubFeature.on_checkpoint_saved`, which calls `write_manifest` to create
`manifest.yaml` in the checkpoint directory -- filling in defaults and overlaying the
fields from `model_hub.manifest` (`id` and `name` are required; here we also set
`description`, `authors`, and `published`). The `published` flag instantly makes the model
visible on the AI Model Hub. This runs on every improvement during training, but
only writes local metadata -- it never uploads anything.

Once the whole training loop finishes, `TorchTrainer.train()` calls
`ModelHubFeature.on_training_end` on the final best checkpoint, which

1. Resolves a backend via `get_backend(model_hub.backend, config)` (`backend: ai-model-hub`
   here) and, depending on `mode`, uploads right away (`online`), checks for internet and
   uploads or defers (`auto`), or always defers and just prints where the checkpoint is
   (`deferred`).
2. With `backend: ai-model-hub`, "upload" means running `itwinai upload-model-to-hub
   <checkpoint_dir>` as a subprocess -- see "Credentials and connectivity" below.

## Part 2: pull and run inference

```bash
itwinai exec-pipeline +pipe-key=inference_pipeline
```

`inference_pipeline` mirrors the same shape: `SyntheticInferenceDatasetGenerator` produces
the inference dataset, fed into `TorchPredictor.execute(inference_dataset)`. `TorchPredictor`
takes `model: nn.Module | ModelLoader` directly, so `model` here is a
`ModelHubModelLoader` `_target_` rather than an already-built model --
`TorchPredictor.execute` detects this (`isinstance(self.model, ModelLoader)`) and calls it
to actually fetch and build the model at runtime.

`ModelHubModelLoader` pulls the checkpoint by `model_id`, auto-discovering
`model.pt` under `root/<checkpoint_dir>/model.pt` (no need to know the exact file path in
the Hub). Because an itwinai checkpoint only contains weights, `model_class` must be given
explicitly. Since `model_class` needs to be the *class itself* (it calls
`self.model_class()` internally) rather than an instance, the config points it at
`synthetic_data.sanity_check_model_class`, a small factory function that returns
`SanityCheckModel` uninstantiated.

By default it pulls from `https://hypha.aicell.io/ri-scale/artifacts`; add a `base_url` key
under the `ModelHubModelLoader` step if you're pointing at a different Hub instance.

`TorchPredictor.execute()` returns a dict mapping each inference item's ID (its index, in
this tutorial's `SyntheticInferenceDataset`) to its predicted value.

## Notes

On `model_class`, if you don't have the original training script available, you can still
inspect the raw `state_dict` to reconstruct a matching class by hand:

```python
state_dict = torch.load(ckpt_path, weights_only=False)
for k, v in state_dict.items():
    print(k, v.shape)
```

## Credentials and connectivity

When `model_hub.backend: ai-model-hub`, pushing a checkpoint (`AIModelHubBackend.upload`)
doesn't talk to the Hub directly -- it shells out to `itwinai upload-model-to-hub
<checkpoint_dir>` as a subprocess. That command is `cli.py`'s `upload_model_to_hub`, which:
downloads `upload_model.py` from GitHub (unless `--upload-script` is given -- which the
subprocess call never passes), and needs a Hub URL and token, resolved with priority
explicit argument > environment variable > `.env` file in the current directory (in this case,
the file was put in the `tutorials/model-hub/torch-tutorial-model-hub`), using
`HYPHA_SERVER_URL` and `HYPHA_TOKEN`. Please see [this page](https://github.com/RI-SCALE/ai-model-hub-example/blob/main/.env.example)
for details on how these parameters should be set.

Because the subprocess call passes neither `--hub-url` nor `--api-token`, **the automatic
push during training only works if `HYPHA_SERVER_URL` and `HYPHA_TOKEN` are already set as
environment variables, or sit in a `.env` file in the working directory `itwinai
exec-pipeline` is run from.**

## Requirements

Internet access is required for both steps: to GitHub and the Hub for Part 1's push, and
to the Hub (`https://hypha.aicell.io/ri-scale/artifacts` by default) for Part 2's pull.
