# Working with Docker containers

This section is intended for the developers of itwinai and outlines the practices used to
manage container images through GitHub Container Registry (GHCR).

## Terminology Recap

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
- `(torch|tf)` is an exclusive OR between "torch" and "tf". You can pick one or the other, but
  not both.
- `x.y` is the **version of the ML framework** (e.g., PyTorch or TensorFlow)
- `distro` is the OS distro in the container (e.g., Ubuntu Jammy)
- `jlab-` is prepended to the tag of images including JupyterLab.

## Image Names and Their Purpose

We use different image names to group similar images under the same namespace:

- **`itwinai`**: Production images. These should be well-maintained and orderly.
- **`itwinai-dev`**: Development images. Tags can vary, and may include random hashes.

Images matching `itwinai:*-latest` will be made available through CVMFS via
[Unpacker](https://gitlab.cern.ch/unpacked/sync).

> [!WARNING] It is very important to keep the number of images matching `itwinai:*-latest` as
> low as possible. Tags should only be created under this namespace when strictly necessary.
> Otherwise, this could cause issues for the Unpacker.

## Building a new container

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

## Docker installation

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
