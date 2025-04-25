# Container image definition files for PyTorch-based itwinai

## Singularity

Example for LUMI (AMD GPUs):

```bash
sudo singularity build --tmpdir /mnt/mbunino/tmp /mnt/mbunino/container_test_2.sif env-files/torch/rocm.def
```

- `/mnt/mbunino/tmp` is a location with enough storage space to support the build.

The Singularity definition file above is bootstrapping from a Singularity image co-developed
by LUMI and AMD, which is mirrored on the interTwin containers registry at
`registry.egi.eu/dev.intertwin.eu/itwinai-dev:lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1`
