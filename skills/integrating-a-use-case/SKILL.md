---
name: integrating-a-use-case
description: Use when adapting existing scientific training code (a standalone PyTorch script, a notebook, or a research repo) into an itwinai plugin - covers scaffolding from the plugin template, porting the training loop, distributed training on HPC, MLflow logging, Ray-based hyperparameter optimization, profiling, and scalability reports. Also use when adding any one of those capabilities to a plugin that already exists.
---

# Integrating a use case into itwinai

Turn existing scientific training code into an installable itwinai plugin that runs distributed
on HPC, logs to MLflow, tunes hyperparameters with Ray, and produces a scalability report.

**This skill targets itwinai 0.4.2.** Phase 0 checks the installed version against it.

## Re-entry: already have a working plugin?

Do not restart from Phase 0. Jump straight to what you need.

| You want to | Go to |
|---|---|
| Start from a training script | Phase 0 |
| Add logging | Phase 4, step 1 |
| Add distributed training | Phase 4, step 2 |
| Add profiling / GPU monitoring | Phase 4, step 3 |
| Add hyperparameter optimization | Phase 4, step 4 |
| Produce a scalability report | Phase 5 |
| Port to another HPC system | `references/slurm.md` |
| Something failed | `references/troubleshooting.md` |

## Rules

1. **Gates are hard.** Each phase ends with a gate. If it fails, go to
   `references/troubleshooting.md` and fix it. Never advance past a failing gate, and never
   report a phase complete without running its gate command and seeing it pass.
2. **Read the source, don't trust this skill for field values.** Every reference names the
   class that owns a set of fields. Read that class from the *installed* itwinai before
   generating config. Users are often on a skill version older than their itwinai.
3. **Load references lazily.** Read a reference file when you reach the step that needs it, not
   before.
4. **Change behaviour and structure in separate steps.** Porting must preserve the science. If
   the ported code trains differently from the original, that is a bug, not an improvement.
5. **Ask before consuming allocation.** Any multi-node job, and anything in Phase 5, needs the
   user's explicit go-ahead.

## Phase 0 - Assess

**Check version skew.**

```bash
itwinai --version
```

If it differs from 0.4.2 above, say so plainly and recommend the user refresh this plugin
(`/plugin` menu) before continuing. Then continue anyway, obeying Rule 2 more strictly.

**Inventory the source code.** Read the training script and write a short table recording where
each of these lives, or "none":

| Concern | Where it is now |
|---|---|
| Model construction | |
| Optimizer / LR scheduler | |
| Loss function | |
| Training loop | |
| Validation loop | |
| Dataset and DataLoader | |
| Metrics | |
| Checkpointing | |
| Hyperparameters (argparse, constants, YAML) | |
| Entry point | |

**Choose the fork.** This decides Phase 2 and nothing else:

- **Components-only** - the model is a plain `nn.Module`, the loss is one of itwinai's built-ins,
  and the loop is a conventional forward/backward/step. You write no trainer at all: a
  `DataGetter` supplies the data and the stock `itwinai.torch.trainer.TorchTrainer` runs it.
  This is the default. Choose it unless something below forces the other branch.
- **Custom trainer** - a `TorchTrainer` subclass. Required when the loss is not in
  `TrainingConfiguration`'s allowed set, the optimizer or scheduler is constructed in a non-standard
  way, the training step is unusual (multiple optimizers, custom backward, physics residuals,
  rollouts), or batches need per-batch transformation.

**Gate 0:** the inventory table is filled in and the fork is chosen and justified in one sentence.

## Phase 1 - Scaffold

Read `references/scaffolding.md`.

Create the plugin repository from `itwinai-plugin-template`, choose the package name, and fix
`pyproject.toml`.

**Gate 1:**

```bash
uv pip install -e .
python -c "import itwinai.plugins.<name>; print('ok')"
```

## Phase 2 - Port

Read `references/porting-training-code.md`, then the branch you chose in Phase 0.

Move the science under `src/itwinai/plugins/<name>/`. Keep the original script around until
Phase 3's gate passes, so you can diff behaviour against it.

**Gate 2:** the import above still succeeds, and no module defines a class whose name collides
with an itwinai class it also imports.

## Phase 3 - Wire

Read `references/pipeline-config.md`.

Write `config.yaml`: a `Pipeline` with steps, hyperparameters at the top, `${}` interpolation to
avoid repeating them.

**Gate 3** - the first rung that actually trains. Use a truncated dataset and one epoch:

```bash
itwinai exec-pipeline --config-name config +pipe_key=training_pipeline epochs=1
```

Run it with no launcher, so itwinai falls back to `NonDistributedStrategy`. The loss must move.
Compare it against a one-epoch run of the original script; they should be in the same range.

## Phase 4 - Capabilities

**Do these in order.** They form a dependency chain, not a menu. `generate-scalability-report`
reads its data out of MLflow, so profiling without logging, or a scaling test without either,
produces an empty report and no error message.

1. **Logging** - `references/logging.md`. Add `LoggersCollection` with `ConsoleLogger` and
   `MLFlowLogger`. Re-run Gate 3 and confirm metrics appear under `mllogs/mlflow`.
2. **Distributed** - `references/distributed.md`, then `references/slurm.md`. Add
   `slurm_config.yaml` and submit a single-node, multi-GPU job.
3. **Profiling** - `references/profiling-and-scalability.md`. Enable `measure_epoch_time` and
   `measure_gpu_data`.
4. **HPO** - `references/hpo.md`. Add the `ray_*` blocks and a search space. Leave `strategy`
   alone; itwinai swaps in the Ray strategy by itself when it detects a Ray cluster.

**Gate 4:** a single-node multi-GPU SLURM job completes, and MLflow contains a run with epoch
times and GPU utilisation recorded.

## Phase 5 - Scale (opt-in)

Only on explicit request. This consumes real allocation and may outlive the session.

Read `references/profiling-and-scalability.md`. Set `mode: scaling-test` and `scalability_nodes`,
submit, wait for every job, then:

```bash
itwinai generate-scalability-report
```

**Gate 5:** the plots exist and are non-empty. Empty plots mean the chain in Phase 4 was broken -
go back to step 1, do not reinterpret the plots.

## Verification ladder

The gates above, as one list. Rungs 1-4 always run; rung 5 is opt-in.

1. Plugin installs and imports (Gate 1, Gate 2)
2. `itwinai sanity-check --torch` passes
3. One-epoch non-distributed run, loss decreases (Gate 3)
4. Single-node multi-GPU SLURM job, metrics in MLflow (Gate 4)
5. Multi-node scaling test, non-empty scalability report (Gate 5)

## Never

- Never write a custom trainer because it feels more thorough. Six of the nine plugins in the
  wild need one; three do not.
- Never copy field lists out of a reference file into config without reading the owning class.
- Never submit a multi-node job without asking.
- Never claim a gate passed without running it.
