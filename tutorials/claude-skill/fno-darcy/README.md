# FNO on Darcy flow — starting point for the Claude skill tutorial

This directory holds the *unintegrated* training code used by the tutorial "Integrating a New
Use Case with the Claude Skill", whose source is at
`docs/tutorials/claude-skill/integrate-a-new-use-case.rst`.

`train.py` trains a Fourier Neural Operator to emulate a 2D Darcy flow solver: it learns the map
from a permeability field `a(x)` to the pressure field `u(x)` solving

```text
-div(a grad u) = f,   u = 0 on the boundary
```

Learning a fast surrogate for an expensive solver is a core digital-twin pattern, which makes
this a more representative example than an image classifier.

## Deliberately plain

The script is written the way scientific training code usually looks *before* integration: one
file, `argparse`, a hand-written training loop, `print` for progress, `torch.save` for
checkpointing. That is the point — the tutorial is about transforming this shape, not about the
neural operator itself.

**Do not tidy this file up.** Its roughness is what the tutorial's Phase 0 inventory reacts to.

## Running it

No dataset download: the permeability fields are sampled as thresholded Gaussian random fields
and the PDE is solved by a matrix-free conjugate-gradient solver in numpy.

```bash
python train.py --epochs 20
```

Only `numpy` and `torch` are required, both of which come with `itwinai[torch]`.

Defaults (256 training samples on a 32x32 grid, 20 epochs) run in well under a minute on CPU.
For a faster check:

```bash
python train.py --epochs 5 --grid-size 16 --n-train 64 --n-val 16
```

The metric is relative L2 error, the standard neural-operator measure. It should fall below
roughly 0.15 within 10 epochs at the default settings.

## Why this example needs a custom trainer

Relative L2 is not among the losses `itwinai.torch.config.TrainingConfiguration` supports, so
the integration overrides `create_model_loss_optimizer` on `TorchTrainer`. Everything else —
the model, optimizer and scheduler — is conventional, so the resulting trainer is three lines.

Most use cases do not need even that. See the tutorial for the alternative path.
