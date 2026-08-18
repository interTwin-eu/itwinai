# Logging

Do this **first** among the Phase 4 capabilities. Profiling writes through the logger, and
`itwinai generate-scalability-report` reads its input back out of MLflow. Nothing downstream
works without it, and nothing downstream complains when it is missing.

## Wire it up

```yaml
      logger:
        _target_: itwinai.loggers.LoggersCollection
        loggers:
          - _target_: itwinai.loggers.ConsoleLogger
            log_freq: 100
          - _target_: itwinai.loggers.MLFlowLogger
            experiment_name: ${experiment_name}
            run_name: ${run_name}
            log_freq: epoch
            log_on_workers: -1
```

`LoggersCollection` fans out to several backends at once, which is what you want: console for
the human watching the job, MLflow for the record.

Available loggers are the `Logger` subclasses in `itwinai/loggers.py` - currently
`ConsoleLogger`, `MLFlowLogger`, `WandBLogger`, `TensorBoardLogger`, `Prov4MLLogger`,
`EmptyLogger`, and `LoggersCollection` itself. Read the constructor of whichever you use;
the shared arguments come from the `Logger` base class.

## The two arguments that cause trouble

- **`log_freq`** - `"epoch"`, `"batch"`, or an integer meaning every N batches. `"batch"` on a
  large dataset produces enormous logs and measurably slows training. Default to `"epoch"`, and
  use an integer for the console logger.
- **`log_on_workers`** - which distributed ranks log. Default is `0`, i.e. rank zero only.
  **The scalability report needs per-worker GPU data, so set `log_on_workers: -1`** (all
  workers) on the MLflow logger before running a scaling test. Leaving it at the default is one
  of the two ways to get an empty report.

## Logging from inside a trainer

Use `self.log(...)`, not the backend directly. It respects `log_freq` and `log_on_workers`, and
it routes to every logger in the collection. Read the `log` method on `TorchTrainer` for the
signature and the supported `kind` values.

Under Ray, also use `self.ray_report(...)` so that Tune sees the metric it is scheduling on.

## Where it lands

By default `mllogs/mlflow` under the launch directory. Inspect it with:

```bash
itwinai mlflow-ui --path mllogs/mlflow
```

Set `tracking_uri` to point at a remote tracking server instead. On HPC, prefer the default
filesystem backend unless a server is already reachable from the compute nodes - a tracking URI
that resolves on the login node but not on the workers fails mid-run.

## Check

After adding the logger, re-run Gate 3 and confirm a run appears with your metrics. Do not
proceed to profiling on the assumption that it worked.
