.. _itwinai_container_usage:

Using itwinai Container Images
==============================

itwinai provides pre-built container images to facilitate the deployment and scaling of machine
learning applications. These containers are built at each release and are available in two
locations:

- **Docker images**: Hosted on GitHub Container Registry (GHCR) under ``ghcr.io/intertwin-eu``
- **Singularity images**: Available on Harbor at ``registry.cern.ch/itwinai``

These images provide a convenient base for users to build their applications on top of a
pre-configured environment optimized for high-performance computing (HPC) and deep learning
workflows.

Available Container Images
--------------------------

Three types of container flavors are available:

- **torch-slim**: Includes Horovod and DeepSpeed, tested on HPC (currently Vega)
- **torch-skinny**: A minimal installation with a significantly reduced image size
- **jlab-slim**: Designed for JupyterLab single-user mode, supports offloading to HPC via
  interLink

Container Registries
--------------------

Docker Containers (GHCR)
~~~~~~~~~~~~~~~~~~~~~~~~

Docker images are hosted on GitHub Container Registry (GHCR) under two repositories:

- **Release Images**: ``ghcr.io/intertwin-eu/itwinai``
- **Development Images**: ``ghcr.io/intertwin-eu/itwinai-dev``

  - Includes images built from pushes to the ``main`` branch.

To use these containers as a base for your application, reference them in your ``Dockerfile``
with the ``FROM`` directive.

All the containers under ``ghcr.io/intertwin-eu/itwinai`` image, whose tag matches ``*-latest``
will be made available through CVMFS via `Unpacker <https://gitlab.cern.ch/unpacked/sync>`_.

Singularity Containers (Harbor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Singularity images are stored on Harbor (``registry.cern.ch``) and can be directly
pulled on HPC systems without conversion. Only release images are converted to Singularity and
pushed to the containers registry.

For JupyterHub integration, use the following URI format to launch JupyterLab single-user
containers::

    oras://registry.cern.ch/itwinai/itwinai:TAG

For example::

    oras://registry.cern.ch/itwinai/itwinai:jlab-slim-latest

Building Applications on Top of itwinai Containers
--------------------------------------------------

Users can build their applications by creating a ``Dockerfile`` that extends an ``itwinai``
container image. Below is an example ``Dockerfile`` using ``torch-slim-latest`` as a base
image:

.. code-block:: dockerfile

    FROM ghcr.io/intertwin-eu/itwinai:torch-slim-latest

    # Set working directory
    WORKDIR /app

    # Copy application dependencies
    COPY requirements.txt ./

    # Install dependencies
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY . .

    # Set entrypoint
    CMD ["python", "main.py"]

This example assumes that:

- The application dependencies are listed in ``requirements.txt``
- The application code (including ``main.py``) is copied into the container
- The application is executed by running ``python main.py``

If you want to create a JupyterLab container you can replace
``ghcr.io/intertwin-eu/itwinai:jlab-slim-latest`` in the ``FROM`` clause above.

By using ``itwinai`` container images, users can focus on their application logic while
leveraging a pre-configured environment optimized for HPC and deep learning workflows.
