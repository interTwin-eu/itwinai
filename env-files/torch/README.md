# Container image definition files for PyTorch-based itwinai

## Singularity

This example is for building the itwinai container for LUMI (AMD GPUs) locally (use `scp` to transfer the final image
to LUMI)

First navigate with `cd` to the base folder of itwinai.

From there, download the singularity base image for pytorch with ROCm:

```bash
singularity pull rocm-base-pytorch.sif REGISTRY_IMG
```

You can choose the following base images:

- `oras://registry.egi.eu/dev.intertwin.eu/itwinai-dev:lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1`
- `oras://registry.cern.ch/itwinai/lumi:lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9`

Other base images can be found on LUMI at `/appl/local/containers/tested-containers` and
`/appl/local/containers/sif-images`. See the
[docs](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#getting-the-container-image)
for more info.

Then build the final container with:

```bash
sudo singularity build --tmpdir /tmp itwinai-lumi-dev.sif env-files/torch/rocm.def
```

- `/tmp` is a location with enough storage space to support the build.

Available itwinai images can be found at:

- `oras://registry.egi.eu/dev.intertwin.eu/itwinai-dev:lumi-itwinai-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1`
- `oras://registry.cern.ch/itwinai/lumi:itwinai0.3.3-lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9`
