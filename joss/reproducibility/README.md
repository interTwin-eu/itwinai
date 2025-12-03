# Reproducing the results from the paper

Here we describe the procedure to reproduce the results shown in the plots of
the paper, concerning the physics use case targeting gravitational-wave analysis
at the [Virgo](https://www.virgo-gw.eu/) interferometer. More details on this
use case can be found [on this
page](https://www.intertwin.eu/article/thematic-module-glitchflow).

Below you will find instructions both for reproducing the experiments on an HPC
system similar to [JUWELS
Booster](https://apps.fz-juelich.de/jsc/hps/juwels/index.html), the one used to
run the experiments shown on the paper, and for *single-host* environments
(e.g., laptop, VM, single HPC node) that are usually accessed via interactive
shells.

We assume that on HPC you have access to a SLURM scheduler, whereas on a
single-host setting you have access to Docker or equivalent (e.g., podman).

## Environment

Below some instruction on how to reuse the same dependencies and itwinai version.

The results in the paper were obtained with itwinai `v0.3.4`, and the same dependencies can be
found by using the following OCI container:
`ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments`

HPC systems generally prefer Apptainer or Singularity over Docker. If on an HPC, first
download the container:

```bash
singularity pull itwinai.sif docker://ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments
```

## Dataset Generation

First we need to generate a synthetic training dataset. For the sake of simplicity, we will
generate a smaller one compared to the one used in the experiments reported in the paper.
Therefore, the results may not be identical.

On an HPC system... TODO

On an single-host setup,

```bash
docker run ghcr.io/intertwin-eu/itwinai:joss-virgo-experiments \
    itwinai exec-pipeline +pipe_key=training_pipeline_small
```
