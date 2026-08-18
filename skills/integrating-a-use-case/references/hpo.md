# Hyperparameter optimization

itwinai runs HPO through Ray Tune. Add four blocks to the trainer step; write no HPO code.

## Do not change `strategy`

Leave it as `ddp`. When itwinai detects a running Ray cluster it substitutes the Ray equivalent
strategy automatically - see `references/distributed.md`. Setting `strategy` to a Ray value by
hand is the single most common mistake when adding HPO, and it is never necessary.

## The four blocks

```yaml
      ray_scaling_config:
        _target_: ray.train.ScalingConfig
        num_workers: ${ray_num_workers}
        use_gpu: true
        resources_per_worker:
          CPU: ${ray_cpus_per_worker}
          GPU: ${ray_gpus_per_worker}

      ray_tune_config:
        _target_: ray.tune.TuneConfig
        num_samples: ${ray_num_trials}
        scheduler:
          _target_: ray.tune.schedulers.ASHAScheduler
          metric: loss
          mode: min
          grace_period: 5
          reduction_factor: 6

      ray_run_config:
        _target_: ray.tune.RunConfig
        storage_path: ${itwinai.cwd:}/ray_checkpoints
        name: FNO-HPO-Experiment

      ray_search_space:
        batch_size:
          type: choice
          categories: [16, 32, 64]
        optim_lr:
          type: uniform
          lower: 1e-5
          upper: 1e-3
```

- **`ray_scaling_config`** - resources per trial. `num_workers` is GPUs per trial, not trials.
  `GPU` may be fractional (below 1.0) to pack several trials onto one GPU, but only with
  `num_workers: 1`.
- **`ray_tune_config`** - `num_samples` is how many trials to run. The scheduler is optional;
  ASHA stops unpromising trials early and is the usual choice.
- **`ray_run_config`** - **`storage_path` must be absolute.** Use `${itwinai.cwd:}`. A relative
  path fails once Ray workers start in different directories.
- **`ray_search_space`** - see below.

## Search space syntax

Each entry becomes `ray.tune.<type>(**rest)`. `type` names the Ray Tune sampler and the
remaining keys are its arguments:

```yaml
        optim_lr:      {type: loguniform, lower: 1e-5, upper: 1e-2}
        batch_size:    {type: choice, categories: [16, 32, 64]}
        num_layers:    {type: randint, lower: 2, upper: 8}
```

The parsing lives in `itwinai/torch/tuning.py`; the samplers are Ray's, documented at
`https://docs.ray.io/en/latest/tune/api/search_space.html`. Already-constructed Tune objects and
`grid_search` dictionaries are passed through untouched.

**The names must match keys the trainer actually reads**, i.e. fields of `TrainingConfiguration`
or of your subclass. A search space over a misspelled name tunes nothing and reports no error.

## Reporting the metric

The scheduler's `metric` must be a metric the trainer reports to Ray. Inside a custom trainer use
`self.ray_report(...)`; read the method on `TorchTrainer` for the signature. If the metric never
arrives, ASHA has nothing to schedule on and every trial runs to completion - which looks like
HPO working, only slowly.

## Running it

HPO needs a Ray cluster. In `slurm_config` set `use_ray: true`; the builder then generates the
cluster startup. The eurac and virgo use cases in the itwinai repository ship `slurm_ray.sh`
scripts that show the shape for reference.

Outside Ray, the `ray_*` blocks are ignored and the trainer logs a warning for each. Seeing
those warnings means no Ray cluster was detected.

## Sizing

Start with `num_samples: 2` and one or two search dimensions, confirm trials appear in MLflow as
nested runs, and only then widen the space. An HPO sweep that is misconfigured is expensive to
discover late.
