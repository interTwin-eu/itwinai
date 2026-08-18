# Profiling and the scalability report

## The dependency chain

The scalability report is not a separate feature. It is a read-back of what logging and
profiling wrote:

```text
MLFlowLogger  ->  measure_epoch_time / measure_gpu_data  ->  mode: scaling-test  ->  generate-scalability-report
```

Break any link and `itwinai generate-scalability-report` produces plots with nothing in them,
and exits successfully. There is no error to read. If the report is empty, walk this chain from
the left; do not reinterpret the plots.

The two most common breaks are a missing `MLFlowLogger`, and `log_on_workers` left at its
default of `0` so that only rank zero contributed GPU data.

## Trainer flags

Set these on the trainer step:

```yaml
      measure_epoch_time: true
      measure_gpu_data: true
      enable_torch_profiling: false
      store_torch_profiling_traces: false
```

Read the `TorchTrainer.__init__` signature in `itwinai/torch/trainer.py` for the full set and
the current defaults.

- **`measure_epoch_time`** - per-epoch wall time. Feeds the epoch-time and speedup plots. Cheap;
  leave it on.
- **`measure_gpu_data`** - GPU utilisation and power draw sampled during the run. Feeds the GPU
  plots. Cheap; leave it on.
- **`enable_torch_profiling`** - the PyTorch profiler. Real overhead and large output. Turn it
  on to diagnose a specific bottleneck, then turn it off. Not needed for the scalability report.
- **`store_torch_profiling_traces`** - keeps the raw traces. Requires `enable_torch_profiling`;
  setting it alone raises `ValueError`.
- **`profiling_wait_epochs` / `profiling_warmup_epochs`** - skip the first epochs, which are
  distorted by warmup and data caching.

For distributed runs also set `log_on_workers: -1` on the MLflow logger, as described in
`references/logging.md`.

## Phase 5: the scaling test

Ask before running this. It submits one job per node count and consumes real allocation.

```yaml
slurm_config:
  mode: scaling-test
  scalability_nodes: "1, 2, 4, 8"
```

Then submit, and **wait for every job to finish** - a report generated while jobs are still
queued silently omits them, which looks like a scaling cliff.

```bash
itwinai generate-scalability-report
```

Useful options - check `itwinai generate-scalability-report --help` for the current list:

- `--experiment-name` and `--run-names` to select what is read
- `--plot-dir` for the output directory
- `--include-communication` for communication fractions. These are unreliable and vary
  substantially between HPC systems; the CLI says so itself. Do not present them as a headline
  result.

Output is a set of plots - absolute and relative epoch time, GPU utilisation and power - written
to `plot_dir`. The generating code is `itwinai/scalability_report/`.

## Reading the result honestly

- Compare like with like: same dataset size, same per-worker batch size, same number of epochs.
- Scaling that looks perfect usually means the job was too small and dominated by fixed costs.
- Scaling that collapses at high node counts is often a data-loading limit, not a communication
  one. `num_workers_dataloader` and the filesystem are the first things to check.

## Python-level profiling

Separate from the above, and useful when a single-node run is slower than it should be:

```yaml
slurm_config:
  py_spy: true
  profiling_sampling_rate: 10
```

Then `itwinai generate-py-spy-report` for the aggregation, or `itwinai generate-flamegraph` for
a flamegraph. See the profiling tutorials in the itwinai documentation.
