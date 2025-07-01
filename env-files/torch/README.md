# Container image definition files for PyTorch-based itwinai

## Singularity

This example is for building the itwinai container for LUMI (AMD GPUs) locally (use `scp` to transfer the final image
to LUMI)

First navigate with `cd` to the base folder of itwinai.

From there, download the singularity base image for pytorch with ROCm:

```bash
singularity pull rocm-base-pytorch.sif oras://registry.egi.eu/dev.intertwin.eu/itwinai-dev:lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1
```

Then build the final container with:

```bash
sudo singularity build --tmpdir /tmp itwinai-lumi-dev.sif env-files/torch/rocm.def
```

- `/tmp` is a location with enough storage space to support the build.

The Singularity definition file above is bootstrapping from the Singularity image (`rocm-base-pytorch.sif`) co-developed
by LUMI and AMD, which is mirrored on the interTwin containers registry at
`registry.egi.eu/dev.intertwin.eu/itwinai-dev:lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1`
