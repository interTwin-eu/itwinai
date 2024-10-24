# JupyterLab image for itwinai with Rucio client

The files in this folder are adapted from the work done by
the [VRE team](https://github.com/vre-hub/environments).

To build this container, go into the root of itwinai and run

```bash
docker build -t <IMG>:<TAG> -f env-files/torch/jupyter/Dockerfile .
```

using your preferred `<IMG>` and `<TAG>`.

## Install custom dependencies

To install custom dependencies (e.g., use cases packages) you can add them
in a `requirements.txt` file, add it somewhere **in the itwinai directory** and pass
it to the `docker build`:

```bash
docker build -t <IMG>:<TAG> -f env-files/torch/jupyter/Dockerfile \
    --build-arg REQUIREMENTS=path/to/requirements.txt .
```

For instance:

```bash
docker build -t <IMG>:<TAG> -f env-files/torch/jupyter/Dockerfile \
    --build-arg REQUIREMENTS=env-files/torch/requirements/cmcc-requirements.txt .
```
