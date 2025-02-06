# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Annotated

import dagger
from dagger import Doc, dag, function, object_type


@object_type
class Singularity:
    """Manage Singularity images."""

    base_image: Annotated[dagger.Container, Doc("Base singularity image")] = (
        dag.container().from_("quay.io/singularity/docker2singularity")
    )
    docker: Annotated[
        dagger.Container | None, Doc("Docker container to convert to Singularity")
    ] = None

    @function
    def client(self) -> dagger.Container:
        """Return base image container with Singularity."""
        return self.base_image

    @function
    def convert(self) -> dagger.File:
        """Export Docker container to a Singularity file."""
        return (
            self.base_image.with_file("img.tar", self.docker.as_tarball())
            .with_exec(["singularity", "build", "img.sif", "oci-archive://img.tar"])
            .file("img.sif")
        )

    @function
    def container(self) -> dagger.Container:
        """Export Docker container to a Singularity file and return a container
        containing the generated SIF.
        """
        return self.client().with_file("container.sif", self.convert())

    @function
    async def publish(
        self,
        password: Annotated[dagger.Secret, Doc("Password for registry")],
        username: Annotated[dagger.Secret, Doc("Username for registry")],
        uri: Annotated[str, Doc("Target URI for the image")],
    ) -> str:
        """Export container and publish it to some registry."""
        print(f"The Singularity image will be published at: {uri}")
        return await (
            self.base_image.with_file("container.sif", self.convert())
            .with_secret_variable(name="SINGULARITY_DOCKER_USERNAME", secret=username)
            .with_secret_variable(name="SINGULARITY_DOCKER_PASSWORD", secret=password)
            .with_exec(["singularity", "push", "container.sif", f"{uri}"])
            .stdout()
        )

    @function
    async def remove(
        self,
        password: Annotated[dagger.Secret, Doc("Password for registry")],
        username: Annotated[dagger.Secret, Doc("Username for registry")],
        registry: Annotated[
            str,
            Doc("The registry URL where the container will be published"),
        ] = "registry.egi.eu",
        project: Annotated[
            str,
            Doc("The name of the project"),
        ] = "dev.intertwin.eu",
        name: Annotated[
            str,
            Doc("The name of the container image"),
        ] = "itwinai-dev",
        tag: Annotated[
            str,
            Doc("Tag for the container image"),
        ] = "latest",
    ) -> str:
        """Remove container from some Harbor registry."""

        cmd = (
            'DIGEST="$(curl -su $SINGULARITY_DOCKER_USERNAME:$SINGULARITY_DOCKER_PASSWORD '
            f'"https://{registry}/api/v2.0/projects/{project}/repositories/{name}/'
            f"artifacts?with_tag=true\" | jq -r '.[] | select(.tags[]?.name == \"'{tag}'\")"
            " | .digest')\""
            "; curl -u $SINGULARITY_DOCKER_USERNAME:$SINGULARITY_DOCKER_PASSWORD -X DELETE "
            f'"https://{registry}/api/v2.0/projects/{project}/repositories/{name}/'
            'artifacts/$DIGEST"'
        )

        return await (
            dag.container()
            .from_("alpine")
            .with_exec(["apk", "add", "curl", "jq"])
            .with_secret_variable(name="SINGULARITY_DOCKER_USERNAME", secret=username)
            .with_secret_variable(name="SINGULARITY_DOCKER_PASSWORD", secret=password)
            .with_exec(["sh", "-c", cmd])
            .stdout()
        )
