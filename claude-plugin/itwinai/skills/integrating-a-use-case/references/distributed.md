# Distributed training

## What you actually change

Usually nothing in the code. If Phase 2 was done properly - using `self.strategy`, `self.device`
and `self.log(...)` instead of raw torch calls and hardcoded `cuda:0` - then distributing the
job is a config and launcher concern.

```yaml
strategy: ddp   # top-level, interpolated into the trainer step
```

Allowed values are the `Literal` on `TorchTrainer.__init__` in `itwinai/torch/trainer.py`:
currently `ddp`, `deepspeed`, `horovod`. Read it rather than trusting this sentence.

- **`ddp`** - the default and the right first choice. Data-parallel, one process per GPU.
- **`deepspeed`** - when the model or optimizer state does not fit in one GPU's memory.
- **`horovod`** - MPI-based, mainly for sites where it is already the supported path. Not
  supported in combination with Ray; itwinai warns and falls back to plain Horovod.

## Strategy resolution is automatic, and this matters

`TorchTrainer._detect_distributed_strategy` decides at runtime, not from config alone:

- **No distributed resources** - falls back to `NonDistributedStrategy` and logs a warning. This
  is what makes Gate 3 runnable on a laptop. "Enough resources" means a detected world size
  greater than one; `ITWINAI_FORCE_DIST=1` overrides the check.
- **A Ray cluster is running** - the Ray equivalent is substituted automatically: `ddp` becomes
  `RayDDPStrategy`, `deepspeed` becomes `RayDeepSpeedStrategy`.

**So do not set `strategy` to a Ray value when adding HPO.** Leave it as `ddp`. The most common
mistake here is "fixing" the strategy field to make HPO work, which is unnecessary.

The corollary is that a run silently training on one GPU is usually a launcher problem, not a
config problem. Check the warning line in the job output before changing config.

## Launching

Do not hand-roll `torchrun`. The SLURM builder generates the correct launcher for the chosen
strategy - see `references/slurm.md`. On HPC that is the supported path, and it is the only one
that stays correct across `ddp`, `deepspeed` and `horovod`.

For a quick interactive check on a multi-GPU node, `itwinai check-distributed-cluster` exercises
the network setup for torch distributed. It expects to be prepended with `torchrun`, or run with
a Ray cluster already up.

## Things that break when going distributed

- **Rank-dependent side effects.** Writing files, downloading data or printing from every rank.
  Guard on `self.strategy.is_main_worker` where the code has such effects.
- **Dataset download in the `DataGetter`.** Every rank runs it. For small datasets this is
  merely wasteful; for anything shared it is a race. Download once outside the job, or guard it.
- **Batch size semantics.** `batch_size` in `config:` is per worker. Total effective batch size
  scales with the number of GPUs, so the learning rate that worked on one GPU frequently does
  not on sixteen. Expect to re-tune, and note this when comparing against the original script.
- **Non-determinism.** Set `random_seed` on the trainer if runs need to be comparable.

## Gate

A single-node, multi-GPU SLURM job that completes and writes metrics to MLflow. Two GPUs is
enough - the point is to prove the distributed path works, not to measure anything.
