# itwinai

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai.assess.sqaaas/main/.report/assessment_output.json)

![itwinai Logo](./docs/images/icon-itwinai-orange-black-subtitle.png)

`itwinai` is a powerful Python toolkit designed to help scientists and researchers streamline AI and machine learning
workflows, specifically for digital twin applications. It provides easy-to-use tools for distributed training,
hyper-parameter optimization on HPC systems, and integrated ML logging, reducing engineering overhead and accelerating
research. Developed primarily by CERN, `itwinai` supports modular and reusable ML workflows, with
the flexibility to be extended through third-party plugins, empowering AI-driven scientific research in digital twins.

See the latest version of our docs [here](https://itwinai.readthedocs.io/).

> [!WARNING]
> Branch protection rules are applied to all branches which names
> match this regex: `[dm][ea][vi]*` . When creating new branches,
> please avoid using names that match that regex, otherwise branch
> protection rules will block direct pushes to that branch.

## Installation

For instructions on how to install `itwinai`, please refer to the
[user installation guide](https://itwinai.readthedocs.io/installation/user_installation.html)
or the
[developer installation guide](https://itwinai.readthedocs.io/installation/developer_installation.html),
depending on whether you are a user or developer

### Docker installation

To build a Docker image for the pytorch version (need to adapt `TAG`):

```bash
# Local
docker buildx build -t itwinai:TAG -f env-files/torch/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:TAG -f env-files/torch/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai:TAG
```

To  build a Docker image for the TensorFlow version (need to adapt `TAG`):

```bash
# Local
docker buildx build -t itwinai:TAG -f env-files/tensorflow/Dockerfile .

# Ghcr.io
docker buildx build -t ghcr.io/intertwin-eu/itwinai:TAG -f env-files/tensorflow/Dockerfile .
docker push ghcr.io/intertwin-eu/itwinai:TAG
```

### Test with `pytest`

Do this only if you are a developer wanting to test your code with pytest.

First, you need to create virtual environments both for torch and tensorflow,
following the instructions above, depending on the system that you are using
(e.g., JSC).

To select the name of the torch and tf environments in which the tests will be
executed you can set the following environment variables.
If these env variables are not set, the testing suite will assume that the
PyTorch environment is under
`.venv-pytorch` and the TensorFlow environment is under `.venv-tf`.

```bash
export TORCH_ENV="my_torch_env"
export TF_ENV="my_tf_env"
```

Functional tests (marked with `pytest.mark.functional`) will be executed under
`/tmp/pytest` location to guarantee isolation among tests.

To run functional tests use:

```bash
pytest -v tests/ -m "functional"
```

> [!NOTE]
> Depending on the system that you are using, we implemented a tailored Makefile
> target to run the test suite on it. Read these instructions until the end!

We provide some Makefile targets to run the whole test suite including unit, integration,
and functional tests. Choose the right target depending on the system that you are using:

Makefile targets:

- Juelich Supercomputer (JSC): `test-jsc`
- In any other case: `test`

For instance, to run the test suite on your laptop user:

```bash
make test
```

## Working with Docker containers

This section is intended for the developers of itwinai and outlines the practices
used to manage container images through GitHub Container Registry (GHCR).

### Terminology Recap

Our container images follow the convention:

```text
ghcr.io/intertwin-eu/IMAGE_NAME:TAG
```

For example, in `ghcr.io/intertwin-eu/itwinai:0.2.2-torch2.6-jammy`:

- `IMAGE_NAME` is `itwinai`
- `TAG` is `0.2.2-torch2.6-jammy`

The `TAG` follows the convention:

```text
[jlab-]X.Y.Z-(torch|tf)x.y-distro
```

Where:

- `X.Y.Z` is the **itwinai version**
- `(torch|tf)` is an exclusive OR between "torch" and "tf". You can pick one or the other, but not both.
- `x.y` is the **version of the ML framework** (e.g., PyTorch or TensorFlow)
- `distro` is the OS distro in the container (e.g., Ubuntu Jammy)
- `jlab-` is prepended to the tag of images including JupyterLab

### Image Names and Their Purpose

We use different image names to group similar images under the same namespace:

- **`itwinai`**: Production images. These should be well-maintained and orderly.
- **`itwinai-dev`**: Development images. Tags can vary, and may include random
hashes.
- **`itwinai-cvmfs`**: Images that need to be made available through CVMFS via
[Unpacker](https://gitlab.cern.ch/unpacked/sync).

> [!WARNING]
> It is very important to keep the number of tags for `itwinai-cvmfs` as low
> as possible. Tags should only be created under this namespace when strictly
> necessary. Otherwise, this could cause issues for the Unpacker.

### Building a new container

Our docker manifests support labels to record provenance information, which can be lately
accessed by `docker inspect IMAGE_NAME:TAG`.

A full example below:

```bash
export BASE_IMG_NAME="what goes after the last FROM"
export IMAGE_FULL_NAME="IMAGE_NAME:TAG"
docker build \
    -t "$IMAGE_FULL_NAME" \
    -f path/to/Dockerfile \
    --build-arg COMMIT_HASH="$(git rev-parse --verify HEAD)" \
    --build-arg BASE_IMG_NAME="$BASE_IMG_NAME" \
    --build-arg BASE_IMG_DIGEST="$(docker pull "$BASE_IMG_NAME" > /dev/null 2>&1 && docker inspect "$BASE_IMG_NAME" --format='{{index .RepoDigests 0}}')" \
    --build-arg ITWINAI_VERSION="$(grep -Po '(?<=^version = ")[^"]*' pyproject.toml)" \
    --build-arg CREATION_DATE="$(date +"%Y-%m-%dT%H:%M:%S%:z")" \
    --build-arg IMAGE_FULL_NAME=$IMAGE_FULL_NAME \ 
    .
```
