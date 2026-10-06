# SLURM and running on HPC

## Use the declarative path

itwinai generates the SLURM script from YAML. Do not hand-write `sbatch` files: hand-written
scripts are the reason most existing plugins cannot run a scaling test, and they have to be
rewritten for every strategy and every cluster.

Only one published plugin (hython) uses the declarative path today. It is nonetheless the
supported one, and it is what makes Phase 5 a config change rather than a rewrite.

## The `slurm_config` block

Add it to the same `config.yaml`, at the top level:

```yaml
slurm_config:
  job_name: fno-darcy
  account: <your-billing-account>
  partition: gpu
  time: "00:30:00"
  num_nodes: 1
  gpus_per_node: 2
  memory: 64G

  mode: single
  distributed_strategy: ${strategy}
  python_venv: .venv

  config_name: config
  config_path: .
  pipe_key: training_pipeline
  run_name: ${run_name}

  submit_job: false
  save_script: true

  pre_exec_file: https://raw.githubusercontent.com/interTwin-eu/itwinai/refs/tags/v0.4.2/src/itwinai/slurm/system-base-scripts/vega_pre_exec.sh

  training_cmd: >
    {itwinai_launcher} exec-pipeline
    --config-name={config_name}
    --config-path={config_path}
    --strategy={distributed_strategy}
    --run-name={run_name}
    +pipe_key={pipe_key}
```

**The schema is `MLSlurmBuilderConfig` in `itwinai/slurm/configuration.py`**, which extends
`SlurmScriptConfiguration`. Read both before writing this block. Every field is documented there
with its default, and that file is the only source of truth for the key names.

Note that the key is `distributed_strategy`. The hython plugin uses `dist_strat`, which does not
match the current schema - do not copy it.

## Curly braces versus `${}`

Two different substitution mechanisms appear in this block and they are not interchangeable:

- `${strategy}` is OmegaConf interpolation, resolved when the config is loaded.
- `{config_name}` inside `training_cmd` is a template slot filled by the SLURM builder from the
  fields of `slurm_config` itself.

Overriding a `{}`-substituted field from the CLI causes a mismatch between the generated command
and the builder's view of it. Change these in the file.

## Running

```bash
itwinai run -c config.yaml      # install plugins, generate the script
itwinai run -jc config.yaml     # ... and submit it
```

`itwinai run` handles plugin installation, SLURM script generation and submission from this one
file. `itwinai generate-slurm` produces the script alone. `itwinai exec-pipeline` is the
low-level command that the generated script ends up calling, and knows nothing about SLURM.

`-j` / `-s` are the only CLI overrides applied on top of `submit_job` and `save_script`. Set
`save_script: true` while developing and read the generated script before submitting - it is the
fastest way to see what the builder actually did.

## `pre_exec_file`

Cluster-specific module loads and environment setup, sourced before the training command.
itwinai ships base scripts for JUWELS, LUMI and Vega under
`src/itwinai/slurm/system-base-scripts/`, usable by path or by URL. For any other system, copy
the closest one and adapt it. This file, the account and the partition are the only parts that
should change when moving a working plugin to a different cluster.

The URL above is pinned to a release tag on purpose. `refs/heads/main` would drift away from the
itwinai this skill targets, and the module stanza is exactly where that hurts. Pin it to the tag
matching the installed itwinai.

**Never invent module names.** The `ml` lines in these scripts are site-specific, and the same
software has different names on every cluster: Vega loads
`CUDA/12.6.0`, `NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0`, `Python/3.12.3-GCCcore-13.3.0`, while
JUWELS loads `Stages/2025 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA`. Guessing produces a job
that dies in the first seconds with a module error. Take the names from the system's own
documentation, from `module avail`, or from a script that already works on that machine, and ask
the user if neither is available. `pre_exec_command` takes a one-liner when a whole file is
overkill; read both fields on `SlurmScriptConfiguration`.

## `mode`

- `single` - one job with the configured strategy. Use this for Gate 4.
- `runall` - one job per strategy, for comparing `ddp`, `deepspeed` and `horovod`.
- `scaling-test` - one job per entry in `scalability_nodes`. Phase 5 only.

## Gate 4

`mode: single`, `num_nodes: 1`, `gpus_per_node: 2`, a short walltime and a truncated dataset.
The job must complete and its metrics must be in MLflow. Ask before submitting anything larger.
