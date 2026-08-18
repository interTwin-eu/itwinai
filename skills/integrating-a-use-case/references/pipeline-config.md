# Writing config.yaml

One YAML file describes the whole workflow: hyperparameters at the top, the pipeline below, and
later the SLURM block. `itwinai exec-pipeline` reads it through Hydra and OmegaConf.

## Shape

```yaml
# Always needed
run_name: fno-darcy
strategy: ddp

# Hyperparameters, referenced below by interpolation
epochs: 20
batch_size: 32
lr: 1e-3
data_root: .tmp/

training_pipeline:
  _target_: itwinai.pipeline.Pipeline
  steps:
    dataloading_step:
      _target_: itwinai.plugins.fno.data.DarcyDataGetter
      data_root: ${data_root}
    training_step:
      _target_: itwinai.plugins.fno.trainer.FNOTrainer
      epochs: ${epochs}
      run_name: ${run_name}
      strategy: ${strategy}
      config:
        batch_size: ${batch_size}
        optim_lr: ${lr}
```

`steps` accepts a mapping or a list. Prefer the mapping: the names appear in logs and make
`itwinai exec-pipeline` overrides readable.

## `_target_`

Every step is instantiated from its fully-qualified import path. For plugin code that is always
`itwinai.plugins.<name>.<module>.<Class>`, and it must be importable from the installed package -
not a path relative to the config file. If a `_target_` fails to resolve, the cause is nearly
always the `pyproject.toml` `include` list, not the config.

Keys nested under a step are constructor arguments, so they must match the class's `__init__`
exactly. `config:` is itself a constructor argument that becomes a `TrainingConfiguration`.

## Interpolation

`${...}` refers to another key in the same file. Declare each hyperparameter once at the top and
interpolate it wherever it is needed, so that a CLI override changes every use at once.

Custom resolvers registered by itwinai:

- `${itwinai.cwd:}` - absolute path of the launch directory
- `${itwinai.range:...}`, `${itwinai.multiply:x,y}`

`${itwinai.cwd:}` exists because several fields, Ray's `storage_path` in particular, reject
relative paths. Read the registrations in `itwinai/cli.py` if you need the current list.

## Running it

```bash
itwinai exec-pipeline --config-name config +pipe_key=training_pipeline
```

- `--config-name` is the filename without `.yaml`; `--config-path` is its directory.
- `+pipe_key=` selects which top-level pipeline to run. The `+` is required because the key is
  being added, not overridden.
- Any field can be overridden Hydra-style: `epochs=1 batch_size=8`.
- `--strategy` and `--run-name` are dedicated flags that create the field if it is missing.

Multiple pipelines can live in one file - `training_pipeline`, `inference_pipeline` - selected by
`pipe_key`. The MNIST use case in the itwinai repository does this.

## Gate 3

```bash
itwinai exec-pipeline --config-name config +pipe_key=training_pipeline epochs=1
```

With no launcher and no Ray cluster, itwinai falls back to `NonDistributedStrategy`, so this
runs anywhere, including a laptop. Truncate the dataset through the `DataGetter` knob rather
than by editing code. The loss must decrease, and it must land in the same range as one epoch of
the original script.
