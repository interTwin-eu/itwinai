from typing import Annotated

import dagger
from dagger import Doc, dag, function, object_type


@object_type
class Singularity:
    """Manage Singularity images."""

    base_image: Annotated[str, Doc("Base singularity image")] = (
        "quay.io/singularity/docker2singularity"
    )

    @function
    def container(
        self,
    ) -> dagger.Container:
        """Return a container with Singularity."""
        return dag.container().from_(self.base_image)

    @function
    def export(
        self,
        container: Annotated[dagger.Container, Doc("Export docker container to singularity")],
    ) -> dagger.File:
        """Export Docker container to a Singularity file."""
        # Credits: https://github.com/shykes/x/blob/main/singularity/main.go
        return (
            self.container()
            .with_file("img.tar", container.as_tarball())
            .with_exec(["singularity", "build", "img.sif", "oci-archive://img.tar"])
            .file("img.sif")
        )

    @function
    async def publish(
        self,
        container: Annotated[
            dagger.Container, Doc("Docker container to convert and publish as SIF")
        ],
        password: Annotated[dagger.Secret, Doc("Password for registry")],
        username: Annotated[dagger.Secret, Doc("Username for registry")],
        uri: Annotated[
            str,
            Doc("Target URI for the image"),
        ],
    ) -> str:
        """Export container and publish it to some registry."""
        print(f"The Singularity image will be published at: {uri}")
        return await (
            self.container()
            .with_file("container.sif", self.export(container=container))
            .with_secret_variable(name="SINGULARITY_DOCKER_USERNAME", secret=username)
            .with_secret_variable(name="SINGULARITY_DOCKER_PASSWORD", secret=password)
            .with_exec(
                [
                    "singularity",
                    "push",
                    "container.sif",
                    f"{uri}",
                ]
            )
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
