# Publishing the trained model

Optional, and independent of the Phase 4 chain: it reads checkpoints, not MLflow. Do it after
Gate 3 at the earliest, once the checkpoints are the ones you actually want to publish.

## What itwinai does for you

`TorchTrainer` holds a `ModelHubFeature` (`itwinai/model_hub/feature.py`). When it is enabled,
the trainer writes a `manifest.yaml` next to each saved checkpoint, and at the end of training
either uploads the best checkpoint or leaves it ready to upload by hand. You write no code.

## Configuration

The feature is configured by a `model_hub` mapping inside the trainer's `config:` block.
`TrainingConfiguration` permits extra fields, which is why it arrives there and is reachable as
`self.config.model_hub`.

```yaml
      config:
        batch_size: ${batch_size}
        optim_lr: ${lr}
        model_hub:
          enabled: true
          mode: deferred
          manifest:
            id: fno-darcy
            name: FNO surrogate for 2D Darcy flow
```

- **`enabled`** - off by default. Nothing is written until you turn it on.
- **`mode`** - `deferred` prepares the checkpoint and stops, `online` uploads at the end of
  training, `auto` uploads only if the machine has a working connection. On HPC the compute
  nodes usually have no outbound network, so `deferred` plus a separate upload is the safe
  choice, and `online` is what silently does nothing useful there.
- **`manifest`** - merged over itwinai's defaults. `id` and `name` are required; the rest
  (license, authors, tags, documentation) is what people will see next to your model. Read
  `write_manifest` in `itwinai/model_hub/manifest.py` for the defaults before inventing keys.

## Uploading

```bash
itwinai upload-model-to-hub <checkpoint-dir>
```

The directory must contain the `manifest.yaml` the trainer wrote, and the command needs the hub
URL and an API token. Pass them through `--env-file`, or export them, and keep the token out of
`config.yaml` and out of git.

## Consuming a published model

`ModelHubModelLoader` (same module) pulls a model by its Model Hub id for inference, instead of
a local checkpoint path. Use it in an inference pipeline, not in the training one.

## Where to read more

`docs/how-it-works/model-hub/explain_model_hub.rst` and
`docs/tutorials/model-hub/model_hub_tutorial.rst` in the itwinai repository.
